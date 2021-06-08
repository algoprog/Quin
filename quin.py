import argparse
import copy
import json
import logging
import math
import os
import pickle
import re
import numpy
import nltk
import torch

from typing import List
from flask import request, Flask, jsonify
from flask_cors import CORS
from scipy import spatial
from sentence_transformers import SentenceTransformer

from retriever.dense_retriever import DenseRetriever
from retriever.sparse_retriever_fast import SparseRetrieverFast
from models.nli import NLI


nltk.download('punkt')
logging.getLogger().setLevel(logging.INFO)


def is_question(query):
    if re.match(r'^(who|when|what|why|which|whose|is|are|was|were|do|does|did|how)', query) or query.endswith('?'):
        return True

    pos_tags = nltk.pos_tag(nltk.word_tokenize(query))
    for tag in pos_tags:
        if tag[1].startswith('VB'):
            return False

    return True


class Quin:
    def __init__(self, docs_file=None, index_path='index', models_path='models/weights',
                 encoder_batch_size=32, nli_batch_size=32, port=1234):
        self.index_path = index_path
        self.encoder_batch_size = encoder_batch_size

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'

        # initialize the sentence tokenizer
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.sent_tokenizer._params.abbrev_types.update(['e.g', 'i.e', 'subsp'])

        # initialize the passage embedding model
        self.text_embedding_model = SentenceTransformer('{}/encoder'.format(models_path),
                                                        device=device)

        if docs_file is None:
            self.nli_model = NLI('{}/nli/nli.state_dict'.format(models_path),
                                 batch_size=nli_batch_size,
                                 device=device)
            self.nli_model.eval()

            if os.path.exists('{}/vectors.pkl'.format(self.index_path)):
                self.dense_index = DenseRetriever(model=self.text_embedding_model, batch_size=32)
                self.dense_index.create_index_from_vectors('{}/vectors.pkl'.format(index_path))
                self.sparse_index = SparseRetrieverFast(path=self.index_path)
                self.documents = pickle.load(open('{}/documents.pkl'.format(index_path), 'rb'))

                self.app = Flask(__name__)
                CORS(self.app)

            self.build_endpoints()
            self.app.run(host='0.0.0.0', port=port)
        else:
            self.index_documents(docs_file=docs_file)

    def index_documents(self, docs_file, sentences_per_snippet=5):
        logging.info('Indexing snippets...')

        self.documents = {}
        all_snippets = []
        with open(docs_file) as f:
            for i, line in enumerate(f):
                document = json.loads(line.rstrip('\n'))
                snippets = self.extract_snippets(document["text"], sentences_per_snippet)
                for snippet in snippets:
                    all_snippets.append(snippet)
                    self.documents[len(self.documents)] = {
                        'snippet': snippet
                    }
                if i % 1000 == 0:
                    logging.info('processed: {} - snippets: {}'.format(i, len(all_snippets)))

        pickle.dump(self.documents, open('{}/documents.pkl'.format(self.index_path), 'wb'))

        logging.info('Building sparse index...')

        self.sparse_index = SparseRetrieverFast(path=self.index_path)
        self.sparse_index.index_documents(all_snippets)

        logging.info('Building dense index...')

        self.dense_index = DenseRetriever(model=self.text_embedding_model,
                                          batch_size=self.encoder_batch_size)
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

    def search(self, query, limit=100):
        """
        Searches the snippet indexes, runs NLI model for statements and highlights relevant sentences
        """
        logging.info('Running sparse retriever for: {}'.format(query))

        sparse_results = self.sparse_index.search([query], topk=limit)[0]
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

        query_is_question = is_question(query)  # Check if the query is a question or statement

        # highlight most relevant sentences
        logging.info('highlighting...')
        results_sentences = []
        sentences_texts = []
        sentences_vectors = {}
        for i, r in enumerate(search_results):
            sentences = self.sent_tokenizer.tokenize(r['snippet'])
            sentences = [s for s in sentences if len(s.split(' ')) > 4]
            sentences_texts.extend(sentences)
            results_sentences.append(sentences)

        vectors = self.text_embedding_model.encode(sentences=sentences_texts, batch_size=128)
        for i, v in enumerate(vectors):
            sentences_vectors[sentences_texts[i]] = v

        query_vector = self.text_embedding_model.encode(sentences=[query], batch_size=1)[0]
        for i, sentences in enumerate(results_sentences):
            best_sentences = set()
            evidence_sentences = []
            for sentence in sentences:
                sentence_vector = sentences_vectors[sentence]
                score = 1 - spatial.distance.cosine(query_vector, sentence_vector)
                if score > 0.9:
                    best_sentences.add(sentence)
                    evidence_sentences.append(sentence)
            if len(evidence_sentences) > 0:
                search_results[i]['evidence'] = ' '.join(evidence_sentences)
            search_results[i]['snippet'] = \
                ' '.join([s if s not in best_sentences else '<b>{}</b>'.format(s) for s in sentences])

        search_results = [s for s in search_results if 'evidence' in s]

        # fact verification
        if not query_is_question:
            logging.info('entailment classification...')
            es_pairs = []
            for result in search_results:
                evidence = result['evidence']
                es_pairs.append((evidence, query))
            try:
                labels, probs = self.nli_model(es_pairs)
                for i in range(len(labels)):
                    confidence = numpy.exp(numpy.max(probs[i]))
                    if confidence > 0.4:
                        search_results[i]['nli_class'] = labels[i]
                    else:
                        search_results[i]['nli_class'] = 'neutral'
                    search_results[i]['nli_confidence'] = str(confidence)
            except:
                search_results = []

        supporting = 0
        refuting = 0
        for r in search_results:
            if 'nli_class' in r:
                if r['nli_class'] == 'entailment':
                    supporting += 1
                elif r['nli_class'] == 'contradiction':
                    refuting += 1

        search_results = search_results[:limit]

        logging.info('done searching')

        if query_is_question:
            return {'type': 'question',
                    'results': search_results}
        else:
            return {'type': 'statement',
                    'supporting': supporting,
                    'refuting': refuting,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quin')
    parser.add_argument('--index',
                        default=None,
                        help='the path to the jsonl docs file to index')
    parser.add_argument('--port',
                        default=1234,
                        help='port to serve the flask API, default: 1234')
    parser.add_argument('--index_path',
                        default='index',
                        help='the path to store the index files, default: index')
    parser.add_argument('--models_path',
                        default='models/weights',
                        help='the path to the model weights, default: models/weights')
    parser.add_argument('--encoder_batch_size',
                        default=32,
                        help='the batch size of the dense encoder')
    parser.add_argument('--nli_batch_size',
                        default=32,
                        help='the batch size of the NLI model')
    args = parser.parse_args()

    q = Quin(docs_file=args.index,
             index_path=args.index_path,
             models_path=args.models_path,
             encoder_batch_size=args.encoder_batch_size,
             nli_batch_size=args.nli_batch_size)
