import logging
import pickle

from retriever.vector_index import VectorIndex


class DenseRetriever:
    def __init__(self, model, batch_size=1, use_gpu=False):
        self.model = model
        self.vector_index = VectorIndex(768)
        self.batch_size = batch_size
        self.use_gpu = use_gpu

    def create_index_from_documents(self, documents):
        logging.info('Building index...')

        self.vector_index.vectors = self.model.encode(documents, batch_size=self.batch_size)
        self.vector_index.build(self.use_gpu)

        logging.info('Built index')

    def create_index_from_vectors(self, vectors_path):
        logging.info('Building index...')
        logging.info('Loading vectors...')
        self.vector_index.vectors = pickle.load(open(vectors_path, 'rb'))
        logging.info('Vectors loaded')
        self.vector_index.build(self.use_gpu)

        logging.info('Built index')

    def search(self, queries, limit=1000, probes=512, min_similarity=0):
        query_vectors = self.model.encode(queries, batch_size=self.batch_size)
        ids, similarities = self.vector_index.search(query_vectors, k=limit, probes=probes)
        results = []
        for j in range(len(ids)):
            results.append([
                (ids[j][i], similarities[j][i]) for i in range(len(ids[j])) if similarities[j][i] > min_similarity
            ])
        return results

    def load_index(self, path):
        self.vector_index.load(path)

    def save_index(self, index_path='', vectors_path=''):
        if vectors_path != '':
            self.vector_index.save_vectors(vectors_path)
        if index_path != '':
            self.vector_index.save(index_path)
