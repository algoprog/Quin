import hashlib
import logging
import math
import multiprocessing
import re
import spacy

from typing import List

from tqdm import tqdm


def get_ngrams(text_tokens: List[str], min_length=1, max_length=4) -> List[str]:
    """
    Gets word-level ngrams from text
    :param text_tokens: the string used to generate ngrams
    :param min_length: the minimum length og the generated ngrams in words
    :param max_length: the maximum length og the generated ngrams in words
    :return: list of ngrams (strings)
    """
    max_length = min(max_length, len(text_tokens))
    all_ngrams = []
    for n in range(min_length - 1, max_length):
        ngrams = [" ".join(ngram) for ngram in zip(*[text_tokens[i:] for i in range(n + 1)])]
        for ngram in ngrams:
            if '#' not in ngram:
                all_ngrams.append(ngram)

    return all_ngrams


def remove_symbols(text):
    text = text.replace('-', ' ')
    text = re.sub('[^a-zA-Z0-9]+', '', text)
    text = ' '.join(text.split())
    return text


def string_hash(string):
    return int(hashlib.md5(string.encode('utf8')).hexdigest(), 16)


def tokenize_function(lemmatization=True, ngrams_length=2, workers=1):
    spacy_obj = spacy.load('en_core_web_sm')
    spacy.prefer_gpu()

    def tokenize(documents):
        # logging.info('Tokenizing {} documents...'.format(len(documents)))
        tokenized_documents = []
        for i, doc in enumerate(tqdm(spacy_obj.pipe(documents,
                                                    disable=["tagger", "parser", "ner"],
                                                    n_threads=workers),
                                     desc='documents',
                                     total=len(documents))):
            tokens = []
            for token in doc:
                if token.is_stop:
                    token = '#'
                else:
                    if lemmatization:
                        token = token.lemma_
                    else:
                        token = token.text
                    token = remove_symbols(token)
                if token != '':
                    tokens.append(token.lower())

            ngrams = get_ngrams(tokens, max_length=ngrams_length)
            tokenized_documents.append(ngrams)

        return tokenized_documents

    return tokenize


class SparseRetriever:
    def __init__(self, ngram_buckets=16777216, k1=1.5, b=0.75, epsilon=0.25):
        self.ngram_buckets = ngram_buckets
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0
        self.inverted_index = {}
        self.idf = {}
        self.doc_len = []
        self.workers = multiprocessing.cpu_count()

    def create_index(self, documents, tokenizer=None):
        spacy.prefer_gpu()

        if tokenizer is not None:
            tokenized_documents = tokenizer(documents)
        else:
            tokenized_documents = documents

        logging.info('Building inverted index...')

        self.corpus_size = len(tokenized_documents)
        nd = self._create_inverted_index(tokenized_documents)
        self._calc_idf(nd)

        logging.info('Built inverted index')

    def _create_inverted_index(self, documents):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for doc_id, document in enumerate(tqdm(documents, desc='documents')):
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1

            for word, freq in frequencies.items():
                hashed_word = string_hash(word) % self.ngram_buckets
                if hashed_word not in self.inverted_index:
                    self.inverted_index[hashed_word] = [(doc_id, freq)]
                else:
                    self.inverted_index[hashed_word].append((doc_id, freq))

            for word, freq in frequencies.items():
                hashed_word = string_hash(word) % self.ngram_buckets
                if hashed_word not in nd:
                    nd[hashed_word] = 0
                nd[hashed_word] += 1

        self.avgdl = num_doc / self.corpus_size

        return nd

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            r_freq = float(freq) / self.corpus_size
            if r_freq > 0.5:
                continue
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def _get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        scores = {}
        for q in query:
            hashed_word = string_hash(q) % self.ngram_buckets
            idf = self.idf.get(hashed_word)
            if idf:
                doc_freqs = self.inverted_index[hashed_word]
                for doc_id, freq in doc_freqs:
                    score = idf * (freq * (self.k1 + 1) /
                                   (freq + self.k1 * 1 - self.b + (self.b * self.doc_len[doc_id] / self.avgdl)))
                    if doc_id in scores:
                        scores[doc_id] += score
                    else:
                        scores[doc_id] = score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def search(self, queries, tokenizer, topk=100):
        # logging.info('Tokenizing queries...')
        tokenized_queries = tokenizer(queries)

        # logging.info('Searching...')
        # t1 = time.time()

        all_results = []
        searched = 0

        for query in tokenized_queries:
            searched += 1
            if searched % 1000 == 0:
                logging.info('{} searches finished'.format(searched))
            results = self._get_scores(query)
            all_results.append(results[:topk])

        # logging.info('Done searching in {}s'.format(time.time() - t1))

        return all_results
