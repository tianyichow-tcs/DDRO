import torch.nn as nn
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=8):
        super().__init__()
        hidden_dims=[2*emb_dim, emb_dim ]
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        layers = []
        input_dim = 2 * emb_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = torch.cat([u, v], dim=1)
        return self.mlp(x).squeeze()
        
class NMFModel(nn.Module):
    def __init__(self, num_users, num_items, k):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=k)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=k)

        self.user_emb.weight.data.uniform_(0., 0.05)
        self.item_emb.weight.data.uniform_(0., 0.05)


    def forward(self, user_idx, item_idx):
        u = F.relu(self.user_emb(user_idx))
        v = F.relu(self.item_emb(item_idx))
        return (u*v).sum(1)


