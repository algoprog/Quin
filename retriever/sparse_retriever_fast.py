import logging
import os
import tantivy

from tqdm import tqdm


class SparseRetrieverFast:
    def __init__(self, path='sparse_index', load=True):
        if not os.path.exists(path):
            os.mkdir(path)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_unsigned_field("doc_id", stored=True)
        schema = schema_builder.build()
        self.index = tantivy.Index(schema, path=path, reuse=load)
        self.searcher = self.index.searcher()

    def index_documents(self, documents):
        logging.info('Building sparse index of {} docs...'.format(len(documents)))
        writer = self.index.writer()
        for i, doc in enumerate(documents):
            writer.add_document(tantivy.Document(
                body=[doc],
                doc_id=i
            ))
            if (i+1) % 100000 == 0:
                writer.commit()
                logging.info('Indexed {} docs'.format(i+1))
        writer.commit()
        logging.info('Built sparse index')
        self.index.reload()
        self.searcher = self.index.searcher()

    def search(self, queries, topk=100):
        results = []
        for q in tqdm(queries, desc='searched'):
            docs = []
            try:
                query = self.index.parse_query(q, ["body"])
                scores = self.searcher.search(query, topk).hits
                docs = [(self.searcher.doc(doc_id)['doc_id'][0], score)
                        for score, doc_id in scores]
            except:
                pass
            results.append(docs)

        return results
