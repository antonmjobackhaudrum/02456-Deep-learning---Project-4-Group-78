from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b

        if type(self) == BM25:
            self.bm25_matrix = self.GetDocumentsBM25Vectors()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    

    def BM25_GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments())
        doc_freq = self.GetDocumentFrequencies()
        IDF = {key: np.log((N + 1) / (value + 0.5)) for key, value in doc_freq.items()}

        return IDF
    
    def GetDocumentsBM25Vectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the bm-25 score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        all_documents = self.index.GetDocuments()

        document_lengths, D_avg = self.GetDocumentLengths()
        bm25_idf = self.BM25_GetInverseDocumentFrequencies()

        for i, doc in enumerate(all_documents):
            f_doc = self.GetDocumentTermCounts(doc)

            for term in f_doc:
                IDF = bm25_idf[term]
                TF_numerator = f_doc[term] * (self.k1 + 1)
                TF_denominator = f_doc[term] + self.k1 * (1 - self.b + self.b * document_lengths[doc] / D_avg)
                TF = TF_numerator / TF_denominator
                bm25_matrix[i, self.term_to_index[term]] = TF * IDF
            
        
        return bm25_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores