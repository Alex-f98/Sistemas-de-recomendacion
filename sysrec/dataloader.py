import torch
import numpy as np
from torch.utils.data import Dataset

class RatingDataset(Dataset):
    def __init__(self, rating_matrix):
        # rating_matrix es un array de NumPy de forma (num_items, num_users)
        self.rating_matrix = rating_matrix
        
        # Obtener índices de ratings no cero
        items, users = np.where(rating_matrix > 0)
        ratings = rating_matrix[items, users]
        
        # Convertir a tensores de PyTorch
        self.users   = torch.tensor(users, dtype=torch.long)
        self.items   = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)
        

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
