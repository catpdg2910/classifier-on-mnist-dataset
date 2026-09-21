import torch
import torch.nn.functional as F
from torch import nn

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 7
        self.patch_embedding = nn.Linear(7*7, 64)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
        self.position_embedding = nn.Parameter(torch.randn(1, 4*4+1, 64))
        encoder_layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 3)
        self.lastfc = nn.Linear(64, 10)
        
    def forward(self, X):
        N, C, H, W = X.shape
        X = X.unfold(2, 7, 7).unfold(3, 7, 7)
        X = X.contiguous().view(N, C, -1, 7, 7)
        X = X.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C*7*7)
        X = self.patch_embedding(X)
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        X = torch.cat((cls_tokens, X), dim = 1)
        X = X + self.position_embedding
        X = self.transformer_encoder(X)
        X = X[:, 0]
        X = self.lastfc(X)
        return X