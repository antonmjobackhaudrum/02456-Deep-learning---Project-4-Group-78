"""
Train the two DPR encoders and save them in checkpoints/.
"""
import os
import hydra
from transformers import AutoTokenizer

from src.data_utils.corpus import Corpus
from src.data_utils.dpr_data import make_dataloader
from src.retrievers.models.encoder import TransformerEncoder
from src.training.trainer import Trainer


@hydra.main(config_path='config', config_name='train_dpr.yaml', version_base=None)
def main(cfg):
    data_dir = (
        f'{cfg.base_path}/data'
        if os.path.exists(f'{cfg.base_path}/data/{cfg.dataset.name}')
        else f'{cfg.base_path}/retrieval_data'
    )
    corpus = Corpus(dataset_name=cfg.dataset.name, base_dir=data_dir)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_loader = make_dataloader(split=corpus.train,
                                   tokenizer=tokenizer,
                                   batch_size=cfg.training.batch_size,
                                   shuffle=True,
                                   max_query_len=cfg.training.max_query_len,
                                   max_doc_len=cfg.training.max_doc_len)
    
    val_loader = make_dataloader(split=corpus.val,
                                   tokenizer=tokenizer,
                                   batch_size=cfg.training.batch_size,
                                   shuffle=False,
                                   max_query_len=cfg.training.max_query_len,
                                   max_doc_len=cfg.training.max_doc_len)

    dropout = cfg.model.get('dropout', 0.1)
    query_encoder = TransformerEncoder(n_layers=cfg.model.n_layers,
                                       n_heads=cfg.model.n_heads,
                                       emb_dim=cfg.model.emb_dim,
                                       hidden_dim=cfg.model.hidden_dim,
                                       feed_forward_dim=cfg.model.feed_forward_dim,
                                       pooling_type=cfg.model.pooling_type,
                                       dropout=dropout)
            
    document_encoder = TransformerEncoder(n_layers=cfg.model.n_layers,
                                          n_heads=cfg.model.n_heads,
                                          emb_dim=cfg.model.emb_dim,
                                          hidden_dim=cfg.model.hidden_dim,
                                          feed_forward_dim=cfg.model.feed_forward_dim,
                                          pooling_type=cfg.model.pooling_type,
                                          dropout=dropout)

    weight_decay = cfg.training.get('weight_decay', 0.01)
    trainer = Trainer(lr=cfg.training.lr, weight_decay=weight_decay, device=cfg.device)
    train_losses, val_losses = trainer.train(query_encoder=query_encoder,
                                                 document_encoder=document_encoder,
                                                 train_loader=train_loader,
                                                 val_loader=val_loader,
                                                 num_epochs=cfg.training.num_epochs,
                                                 checkpoint_dir=cfg.training.checkpoint_dir)

    print(f'Final train loss: {train_losses[-1]:.4f}   final val loss: {val_losses[-1]:.4f}')


if __name__ == '__main__':
    main()
