"""
Training loop for DPR, based on the course notebooks (week 4, 6 and 7).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from src.training.loss import DPRLoss


class Trainer:
    def __init__(self, lr: float = 3e-4, device: str = 'cpu'):
        self.criterion = DPRLoss()
        self.lr = lr
        self.device = device

    def train(self, query_encoder: nn.Module, document_encoder: nn.Module,
              train_loader, val_loader, num_epochs: int, checkpoint_dir: str = 'checkpoints'):
        self.query_encoder = query_encoder.to(self.device)
        self.document_encoder = document_encoder.to(self.device)

        # One optimizer for both encoders
        self.optimizer = torch.optim.AdamW(
            list(self.query_encoder.parameters()) + list(self.document_encoder.parameters()),
            lr=self.lr,
        )

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            batch_loss = []
            self.query_encoder.train()
            self.document_encoder.train()

            for batch in train_loader:
                batch_loss.append(self.train_step(batch))

            train_loss = np.mean(batch_loss)
            val_loss = self.evaluate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch {epoch + 1}/{num_epochs}   train loss: {train_loss:.4f}   val loss: {val_loss:.4f}')

            # Only save when the val loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.query_encoder.state_dict(), os.path.join(checkpoint_dir, 'query_encoder.pt'))
                torch.save(self.document_encoder.state_dict(), os.path.join(checkpoint_dir, 'document_encoder.pt'))
                print(f'             saved new best checkpoint to {checkpoint_dir}/')

        return train_losses, val_losses

    def train_step(self, batch) -> float:
        batch = self._to_device(batch)

        q_vectors = self.query_encoder(batch["query_ids"],batch["query_mask"])
        d_vectors = self.document_encoder(batch["doc_ids"],batch["doc_mask"])
        loss = self.criterion(q_vectors, d_vectors)

        # Course Week 4
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, loader) -> float:
        # Course Week 4
        batch_loss = []
        self.query_encoder.eval()
        self.document_encoder.eval()

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)

                q_vectors = self.query_encoder(batch["query_ids"],batch["query_mask"])
                d_vectors = self.document_encoder(batch["doc_ids"],batch["doc_mask"])
                loss = self.criterion(q_vectors, d_vectors)

                batch_loss.append(loss.item())

        self.query_encoder.train()
        self.document_encoder.train()

        return np.mean(batch_loss)

    def _to_device(self, batch: dict) -> dict:
        return {key: value.to(self.device) for key, value in batch.items()}
