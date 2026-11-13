"""
Dense Passage Retrieval: encode all documents once, then score queries against them.
"""
import torch
from transformers import AutoTokenizer

from src.retrievers.base_retriever import Retriever
from src.retrievers.models.encoder import TransformerEncoder


class DPR(Retriever):
    def __init__(self, cfg, documents: list[dict] = None, index_path: str = None,
                 batch_size: int = 64, max_query_len: int = 64, max_doc_len: int = 256):
        super().__init__(documents, index_path)
        self.cfg = cfg
        self.device = cfg.get('device', 'cpu')
        self.batch_size = batch_size
        # Same max lengths as in training
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.__load_models()

        # The vector database
        self.document_embeddings = self.EncodeDocuments()

    def __load_models(self):
        # n_heads, n_layers, emb_in, hidden_dim, feed_forward_dim
        query_encoder = TransformerEncoder(n_layers=self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)

        document_encoder = TransformerEncoder(n_layers=self.cfg.model.n_layers,
                                           n_heads=self.cfg.model.n_heads,
                                           emb_dim=self.cfg.model.emb_dim,
                                           hidden_dim=self.cfg.model.hidden_dim,
                                           feed_forward_dim=self.cfg.model.feed_forward_dim,
                                           pooling_type=self.cfg.model.pooling_type)


        # Need to read dict (c.f. trainer.py)
        query_weights_path = self.cfg.model.get('query_encoder_weights_path', None)
        if self.cfg.model.get('query_encoder_weights_path', None):
            query_encoder.load_state_dict(torch.load(query_weights_path,map_location=self.device))

        document_weights_path = self.cfg.model.get('document_encoder_weights_path', None)
        if self.cfg.model.get('document_encoder_weights_path', None):
            document_encoder.load_state_dict(torch.load(document_weights_path,map_location=self.device))

        self.query_encoder = query_encoder.to(self.device).eval()
        self.document_encoder = document_encoder.to(self.device).eval()

    @torch.no_grad()
    def _encode(self, encoder: torch.nn.Module, texts: list[str], max_len: int) -> torch.Tensor:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            tokenizer_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
            batch_texts_ = self.tokenizer(batch_texts,max_length=max_len,**tokenizer_args)
            input_ids = batch_texts_["input_ids"].to(self.device)
            attention_mask = batch_texts_["attention_mask"].to(self.device)
            batch_vectors = encoder(input_ids,attention_mask)
            batch_vectors = batch_vectors.cpu()
            vectors.append(batch_vectors)

        return torch.cat(vectors,dim=0)

    def EncodeDocuments(self) -> torch.Tensor:
        documents_text = [document.GetText() for document in self.index.GetDocuments()]
        documents_encoded = self._encode(self.document_encoder,documents_text,self.max_doc_len)

        return documents_encoded

    def CalculateScores(self, queries: list[str]):
        queries_encoded = self._encode(self.query_encoder,queries,self.max_query_len)
        # Same dot product as in DPRLoss
        scores = queries_encoded@self.document_embeddings.T

        return scores.numpy()
