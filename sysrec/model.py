
import torch
import torch.nn as nn
import torch.optim as optim

import dataloader

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, bias=True, global_mean=None):
        super().__init__()
        b=1 if bias else 0
        self.user_emb = nn.Embedding(num_users, latent_dim + b)
        self.item_emb = nn.Embedding(num_items, latent_dim + b)

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        self.global_mean = global_mean
        self._history = []

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=1)

    
    def fit(self, train_data, epochs=100, lr=1e-3, reg=1e-4, history=False):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        total_batch = len(train_data)

        for epoch in range(epochs):
            avg_loss = 0.0

            for items, users, ratings in train_data: #Acá itera por mini-batach, para ser similar a TPS poner full-batch.

                users = users.long()
                items = items.long()
                ratings = ratings.float()
   
                pred = self(users, items)
                loss = ((pred - ratings)**2).mean()
                loss += reg * (self.user_emb.weight.norm(2)**2 +
                               self.item_emb.weight.norm(2)**2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() / total_batch
                if history: self._history.append(loss.item())

            print(f"Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}")

    def predict(self, users, items, p=0.5):
        """
        users, items: tensores de índices
        p: peso para MF puro
        
        Devuelve:
        p * rating_MF  + (1-p) * global_mean
        """
        mf_pred = self.forward(users, items)
        if self.global_mean is None:
            return mf_pred

        return p * mf_pred + (1 - p) * self.global_mean

    def history(self):
        return self._history



