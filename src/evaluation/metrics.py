from src.data_utils.query import Query

def calculateRecall(relevancies: list[bool], query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculatePrecision(relevancies: list[bool]):
    return sum(relevancies) / len(relevancies)

def calculateMRR(relevancies: list[bool]):

    for i in range(len(relevancies)):
        if relevancies[i] == 1:
            return 1/(i+1)

    return 0


def Evaluate(relevancies: list[list[bool]], queries: list[Query]):
    assert len(relevancies) == len(queries), 'The number of relevant passages must match the number of queries'
    scores = {'recall': None, 'precision': None, 'MRR': None}
    n_queries = len(queries)
    total_recall = 0
    total_precision = 0
    total_MRR = 0
    
    for i in range(n_queries):
        total_recall += calculateRecall(relevancies[i], queries[i])
        total_precision += calculatePrecision(relevancies[i])
        total_MRR += calculateMRR(relevancies[i])
    
    scores['recall'] = total_recall / n_queries
    scores['precision'] = total_precision / n_queries
    scores['MRR'] = total_MRR / n_queries
    
    return scores
