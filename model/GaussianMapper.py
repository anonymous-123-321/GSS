import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMapper2(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=64, dropout=0.1):
        super().__init__()

        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) 

        
        
        
        self.mu_ln = nn.LayerNorm(hidden_dim)
        self.sigma_ln = nn.LayerNorm(hidden_dim)

        
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.sigma_head = nn.Linear(hidden_dim, output_dim)
        self.softplus = nn.Softplus()

        
        
        
        nn.init.constant_(self.sigma_head.bias, 2.0)
        nn.init.xavier_uniform_(self.sigma_head.weight, gain=0.01)

    def forward(self, x):
        
        h1 = self.drop1(self.act1(self.ln1(self.fc1(x))))

        h2 = self.drop2(self.act2(self.ln2(self.fc2(h1))))

        feat = h1 + h2

        
        mu = self.mu_head(self.mu_ln(feat))
        sigma = self.softplus(self.sigma_head(self.sigma_ln(feat))) + 1e-6

        return mu, sigma