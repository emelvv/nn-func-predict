
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1) 
