import torch
import torch.nn as nn
import numpy as np

from .embedding import Embedding

class MLP(nn.Module):
    def __init__(self, n_freq=6, d_hidden=128, d_out=1, n_hidden=3, skip_in=[], use_float16=False):
        super().__init__()
        self.emb = Embedding(3, n_freq)
        layers = [
            nn.Linear(self.emb.out_channels, d_hidden),
            nn.Softplus(beta=100)
        ]
        count = 2
        self.skip_count = []
        self.skip_in = skip_in
        for i in range(n_hidden):
            if i in skip_in:
                layers.append(nn.Linear(d_hidden + self.emb.out_channels, d_hidden))
                self.skip_count.append(count)
            else:
                layers.append(nn.Linear(d_hidden, d_hidden))
            count += 1
            layers.append(nn.Softplus(beta=100))
            count += 1
        layers.append(nn.Linear(d_hidden, d_out))
        count += 1
        self.net = nn.ModuleList(layers)
        self.use_float16 = use_float16
    
    def forward(self, x):
        emb = self.emb(x)
        x = emb
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_float16):
            for i, module in enumerate(self.net):
                if i in self.skip_count:
                    x = module(torch.cat([x, emb], dim=-1))
                else:
                    x = module(x)
        return x