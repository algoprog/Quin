import logging
import math
import pickle
import faiss

import numpy as np


class VectorIndex:
    def __init__(self, d):
        self.d = d
        self.vectors = []
        self.index = None

    def add(self, v):
        self.vectors.append(v)

    def build(self, use_gpu=False):
        self.vectors = np.array(self.vectors)

        faiss.normalize_L2(self.vectors)

        logging.info('Indexing {} vectors'.format(self.vectors.shape[0]))

        if self.vectors.shape[0] > 50000:
            num_centroids = 8 * int(math.sqrt(math.pow(2, int(math.log(self.vectors.shape[0], 2)))))

            logging.info('Using {} centroids'.format(num_centroids))

            self.index = faiss.index_factory(self.d, "IVF{}_HNSW32,Flat".format(num_centroids))

            ngpu = faiss.get_num_gpus()
            if ngpu > 0 and use_gpu:
                logging.info('Using {} GPUs'.format(ngpu))

                index_ivf = faiss.extract_index_ivf(self.index)
                clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(self.d))
                index_ivf.clustering_index = clustering_index

            logging.info('Training index...')

            self.index.train(self.vectors)
        else:
            self.index = faiss.IndexFlatL2(self.d)
            if faiss.get_num_gpus() > 0 and use_gpu:
                self.index = faiss.index_cpu_to_all_gpus(self.index)

        logging.info('Adding vectors to index...')

        self.index.add(self.vectors)

    def load(self, path):
        self.index = faiss.read_index(path)

    def save(self, path):
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), path)

    def save_vectors(self, path):
        pickle.dump(self.vectors, open(path, 'wb'), protocol=4)

    def search(self, vectors, k=1, probes=1):
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        try:
            self.index.nprobe = probes
        except:
            pass
        distances, ids = self.index.search(vectors, k)
        similarities = [(2-d)/2 for d in distances]
        return ids, similarities
