import hashlib
import logging
import math
import re

from multiprocessing import Pool
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet, stopwords
from typing import List
from tqdm import tqdm

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))


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


def clean_text(text):
    """
    Removes non-alphanumeric symbols from text
    :param text:
    :return: clean text
    """
    text = text.replace('-', ' ')
    text = re.sub('[^a-zA-Z0-9 ]+', '', text)
    text = re.sub(' +', ' ', text)
    return text


def string_hash(string):
    """
    Returns a static hash value for a string
    :param string:
    :return:
    """
    return int(hashlib.md5(string.encode('utf8')).hexdigest(), 16)


def tokenize(text, lemmatize=True, ngrams_length=2):
    """
    :param text:
    :param stopwords: set of stopwords to exclude
    :param lemmatize:
    :param ngrams_length: the maximum number of tokens per ngram
    :return:
    """
    tokens = clean_text(text).lower().split(' ')
    tokens = [t for t in tokens if t != '']
    if lemmatize:
        lemmatized_words = []
        pos_labels = pos_tag(tokens)
        pos_labels = [pos[1][0].lower() for pos in pos_labels]

        for i, word in enumerate(tokens):
            if word in stopwords:
                word = '#'
            if pos_labels[i] == 'j':
                pos_labels[i] = 'a'  # 'j' <--> 'a' reassignment
            if pos_labels[i] in ['r']:  # For adverbs it's a bit different
                try:
                    lemma = wordnet.synset(word + '.r.1').lemmas()[0].pertainyms()[0].name()
                except:
                    lemma = word
                lemmatized_words.append(lemma)
            elif pos_labels[i] in ['a', 's', 'v']:  # For adjectives and verbs
                lemmatized_words.append(lemmatizer.lemmatize(word, pos=pos_labels[i]))
            else:  # For nouns and everything else as it is the default kwarg
                lemmatized_words.append(lemmatizer.lemmatize(word))
        tokens = lemmatized_words

    ngrams = get_ngrams(tokens, max_length=ngrams_length)

    return ngrams


class SparseRetriever:
    def __init__(self, ngram_buckets=16777216, k1=1.5, b=0.75, epsilon=0.25,
                 max_relative_freq=0.5, workers=4):
        self.ngram_buckets = ngram_buckets
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.max_relative_freq = max_relative_freq
        self.corpus_size = 0
        self.avgdl = 0

        self.inverted_index = {}
        self.idf = {}
        self.doc_len = []
        self.workers = workers

    def index_documents(self, documents):
        with Pool(self.workers) as p:
            tokenized_documents = list(tqdm(p.imap(tokenize, documents), total=len(documents), desc='tokenized'))

        logging.info('Building inverted index...')

        self.corpus_size = len(tokenized_documents)
        nd = self._create_inverted_index(tokenized_documents)
        self._calc_idf(nd)

        logging.info('Built inverted index')

    def _create_inverted_index(self, documents):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for doc_id, document in enumerate(tqdm(documents, desc='indexed')):
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
            if r_freq > self.max_relative_freq:
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
        query = tokenize(query)
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

    def search(self, queries, topk=100):
        results = [self._get_scores(q) for q in tqdm(queries, desc='searched')]
        results = [r[:topk] for r in results]
        logging.info('Done searching')
        return results
