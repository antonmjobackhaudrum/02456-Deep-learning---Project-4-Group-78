from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
import re
from scipy.sparse import lil_matrix


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = text.split()

        return text
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        unique_words = self.PreprocessText(query)
        unique_words = set(unique_words)
        return list(unique_words)

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        document_text = self.PreprocessText(document.GetText())
        unique_words_doc = set(document_text)
        return list(unique_words_doc)
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        df = {}

        all_documents = self.index.GetDocuments()
        for doc in all_documents:
            for term in self.GetDocumentVocabulary(doc):
                if term in df:
                    df[term] += 1
                else:
                    df[term] = 1

        return df
       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        N = len(self.index.GetDocuments())
        doc_freq = self.GetDocumentFrequencies()
        IDF = {key: np.log(N / value) for key, value in doc_freq.items()}

        return IDF
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        vocabulary = self.GetDocumentVocabulary(document)
        document_text = self.PreprocessText(document.GetText())
        df = {word: 0 for word in vocabulary}
        
        for term in vocabulary:
            for word in document_text:
                if word == term:
                    df[term] += 1

        return df

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        all_documents = self.index.GetDocuments()
        for i, doc in enumerate(all_documents):
            f_doc = self.GetDocumentTermCounts(doc)
            f_max = max(f_doc.values())

            for term in f_doc:
                IDF = self.idf[term]
                TF = f_doc[term] / f_max
                tfidf_matrix[i, self.term_to_index[term]] = TF * IDF
            
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """

        if isinstance(queries, str):
            queries = [queries]
        
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for i, query in enumerate(queries):
            text = self.GetQueryVocabulary(query)
            for term in text:
                if term in self.term_to_index:
                    j = self.term_to_index[term]
                    query_vector[i,j] += 1
            
        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    