"""
Evaluate TF-IDF, BM25 or DPR, e.g. python main.py retriever=bm25
"""
import hydra
from src.data_utils.corpus import Corpus
from src.data_utils.query import Query
from src.retrievers.tf_idf import TFIDF
from src.retrievers.bm25 import BM25
from src.retrievers.dpr import DPR
from src.retrievers.utils import retrieveQueryAndGetRelevancies
from src.evaluation.metrics import Evaluate


def make_queries(corpus: Corpus, split) -> list[Query]:
    return [Query(split['question'][idx],
                  id=split['query_id'][idx],
                  relevant_document_ids=corpus.relevant_ids[split['query_id'][idx]])
            for idx in range(split.num_rows)]


def build_retriever(cfg, corpus: Corpus):
    method = cfg.retriever.method

    if method == "tf_idf":
        return TFIDF(documents=corpus.all_documents)

    if method == "bm25":
        return BM25(documents=corpus.all_documents)

    if method == "dpr":
        return DPR(cfg=cfg,documents=corpus.all_documents)

    raise ValueError(f'Unknown retriever method: {method}')


@hydra.main(config_path='config', config_name='fiqa_experiment.yaml', version_base=None)
def main(cfg):
    ### Load a dataset
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=f'{cfg.base_path}/retrieval_data')

    ### Prepare queries
    splits = {'Train': corpus.train, 'Val': corpus.val, 'Test': corpus.test}
    queries = {name: make_queries(corpus, split) for name, split in splits.items() if split is not None}

    ### Initialize retriever
    retriever = build_retriever(cfg, corpus)
    print(f'Retriever: {cfg.retriever.method}')

    for name, split_queries in queries.items():
        ### Retrieve top-k relevant queries
        retriever_relevancies = retrieveQueryAndGetRelevancies(retriever, split_queries, k=cfg.top_k)

        ### Evaluate
        scores = Evaluate(retriever_relevancies, split_queries)

        print(f'{name} results: {scores}')


if __name__ == '__main__':
    main()
