import copy
import json
import logging
import math
import os
import pickle
import numpy
import nltk
import torch

from typing import List
from flask import request, Flask, jsonify
from flask_cors import CORS
from scipy import spatial
from sentence_transformers import SentenceTransformer

from retriever.dense_retriever import DenseRetriever
from retriever.sparse_retriever import SparseRetriever, tokenize_function
from models.nli import NLI
from models.passage_ranker import PassageRanker
from utils import is_question, softmax

nltk.download('punkt')

logging.getLogger().setLevel(logging.INFO)


class Quin:
    def __init__(self, index_path='index'):
        self.index_path = index_path

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._params.abbrev_types.update(['e.g', 'i.e', 'subsp'])

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        self.text_embedding_model = SentenceTransformer('models/weights/encoder',
                                                        device=device)
        self.passage_ranking_model = PassageRanker(model_path='models/weights/passage_ranker/passage_ranker.state_dict',
                                                   batch_size=32,
                                                   device=device)
        self.nli_model = NLI('models/weights/nli/nli_model.state_dict',
                             batch_size=32,
                             device=device)

        self.tokenizer = tokenize_function()

        if os.path.exists('{}/vectors.pkl'.format(self.index_path)):
            self.dense_index = DenseRetriever(model=self.text_embedding_model, batch_size=32)
            self.dense_index.create_index_from_vectors('{}/vectors.pkl'.format(index_path))
            self.sparse_index = pickle.load(open('{}/sparse_index.pkl'.format(index_path), 'rb'))
            self.documents = pickle.load(open('{}/documents.pkl'.format(index_path), 'rb'))

        self.app = Flask(__name__)
        CORS(self.app)

        logging.info('Initialized!')

    def index_documents(self, documents: List[str], batch_size=32, sentences_per_snippet=5):
        logging.info('Indexing snippets...')

        self.documents = {}
        all_snippets = []
        for i, document in enumerate(documents):
            snippets = self.extract_snippets(document, sentences_per_snippet)
            for snippet in snippets:
                all_snippets.append(snippet)
                self.documents[len(self.documents)] = {
                    'snippet': snippet
                }

            if i % 1000 == 0:
                logging.info('processed: {} - snippets: {}'.format(i, len(all_snippets)))

        pickle.dump(self.documents, open('{}/documents.pkl'.format(self.index_path), 'wb'))

        logging.info('Building sparse index...')

        tokenizer = tokenize_function()
        self.sparse_index = SparseRetriever()
        self.sparse_index.create_index(all_snippets, tokenizer=tokenizer)
        pickle.dump(self.sparse_index, open('{}/sparse_index.pkl'.format(self.index_path), 'wb'))

        logging.info('Building dense index...')

        self.dense_index = DenseRetriever(model=self.text_embedding_model,
                                          batch_size=batch_size)
        self.dense_index.create_index_from_documents(all_snippets)
        self.dense_index.save_index(vectors_path='{}/vectors.pkl'.format(self.index_path))

        logging.info('Done')

    def extract_snippets(self, text, sentences_per_snippet=5):
        """ Extracts snippets from text with a sliding window """
        sentences = self.sent_tokenizer.tokenize(text)
        snippets = []
        i = 0
        last_index = 0
        while i < len(sentences):
            snippet = ' '.join(sentences[i:i + sentences_per_snippet])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
            last_index = i + sentences_per_snippet
            i += int(math.ceil(sentences_per_snippet / 2))
        if last_index < len(sentences):
            snippet = ' '.join(sentences[last_index:])
            if len(snippet.split(' ')) > 4:
                snippets.append(snippet)
        return snippets

    def search(self, query, limit=1000, highlight=True, min_relevance=0.3):
        """
        Searches the snippet indexes, runs NLI model for statements and highlights relevant sentences
        """
        logging.info('Running sparse retriever for: {}'.format(query))

        sparse_results = self.sparse_index.search([query], tokenizer=self.tokenizer, topk=limit)[0]
        sparse_results = [r[0] for r in sparse_results]

        logging.info('Running dense retriever for: {}'.format(query))

        dense_results = self.dense_index.search([query], limit=limit)[0]
        dense_results = [r[0] for r in dense_results]

        results = list(set(sparse_results + dense_results))

        search_results = []
        if len(results) > 0:
            for i in range(len(results)):
                doc_id = results[i]
                result = copy.copy(self.documents[doc_id])
                search_results.append(result)

        # Re-rank the results using a binary relevance classifier
        snippets = [s['snippet'] for s in search_results]
        qa_pairs = [(query, snippet) for snippet in snippets]
        _, probs = self.passage_ranking_model(qa_pairs)
        probs = [softmax(p)[1] for p in probs]
        filtered_results = []
        for i in range(len(search_results)):
            if probs[i] > min_relevance:
                search_results[i]['score'] = probs[i]
                filtered_results.append(search_results[i])
        search_results = filtered_results

        query_is_question = is_question(query)  # Check if the query is a question or statement

        # If the query is a statement, run an NLI model to classify the results in
        # the 3 following classes: entailment, contradiction, neutral
        if not query_is_question:
            es_pairs = []
            for result in search_results:
                evidence = result['snippet']
                es_pairs.append((evidence, query))
            labels, probs = self.nli_model(es_pairs)
            for i in range(len(labels)):
                confidence = numpy.exp(numpy.max(probs[i]))
                search_results[i]['nli_class'] = labels[i]
                search_results[i]['nli_confidence'] = str(confidence)

        search_results = sorted(search_results, key=lambda x: x['score'], reverse=True)
        search_results = search_results[:limit]

        for i, result in enumerate(search_results):
            search_results[i]['score'] = str(search_results[i]['score'])

        if highlight:  # highlight the most relevant sentences
            results_sentences = []
            sentences_texts = []
            sentences_vectors = {}
            for i, r in enumerate(search_results):
                sentences = self.sent_tokenizer.tokenize(r['snippet'])
                sentences = [s for s in sentences if len(s.split(' ')) > 4]
                sentences_texts.extend(sentences)
                results_sentences.append(sentences)

            vectors = self.text_embedding_model.encode(sentences=sentences_texts, batch_size=64)
            for i, v in enumerate(vectors):
                sentences_vectors[sentences_texts[i]] = v

            query_vector = self.text_embedding_model.encode(sentences=[query], batch_size=1)[0]
            for i, sentences in enumerate(results_sentences):
                best_sentences = set()
                for sentence in sentences:
                    sentence_vector = sentences_vectors[sentence]
                    score = 1 - spatial.distance.cosine(query_vector, sentence_vector)
                    if score > 0.88:
                        best_sentences.add(sentence)

                search_results[i]['snippet'] = \
                    ' '.join([s if s not in best_sentences else '<b>{}</b>'.format(s) for s in sentences])

        return {'type': 'question' if query_is_question else 'statement',
                'results': search_results}

    def build_endpoints(self):
        @self.app.route('/search', methods=['POST', 'GET'])
        def search_endpoint():
            query = request.args.get('query').lower()
            limit = request.args.get('limit') or 100
            limit = min(int(limit), 1000)
            results = json.dumps(self.search(query,
                                             limit=limit), indent=4)
            return results

    def serve(self, port=80):
        self.build_endpoints()
        self.app.run(host='0.0.0.0', port=port)


q = Quin(index_path='index')
q.index_documents(documents=[
    'Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with his wife, Emma Thomas. The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets.',
    'Interstellar is a 2014 epic science fiction film directed, co-written and produced by Christopher Nolan. It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.'
])
q.serve()
