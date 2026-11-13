"""
DataLoader with (question, relevant document) pairs for training DPR.
"""
import random

from torch.utils.data import DataLoader


class PairCollator:
    def __init__(self, tokenizer, max_query_len: int = 64, max_doc_len: int = 256):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch):
        questions = [row['question'] for row in batch]
        # Pick one of the relevant documents at random, so a question is only once in each batch
        documents = [random.choice(row['text']) for row in batch]

        tokenizer_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        questions_ = self.tokenizer(questions,max_length=self.max_query_len,**tokenizer_args)
        documents_ = self.tokenizer(documents,max_length=self.max_doc_len,**tokenizer_args)

        return {'query_ids': questions_['input_ids'],
                'query_mask': questions_['attention_mask'],
                'doc_ids': documents_['input_ids'],
                'doc_mask': documents_['attention_mask']}



def make_dataloader(split, tokenizer, batch_size: int, shuffle: bool = True,
                    max_query_len: int = 64, max_doc_len: int = 256) -> DataLoader:
    return DataLoader(
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairCollator(tokenizer, max_query_len, max_doc_len),
    )
