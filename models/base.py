import torch.nn as nn
import torch


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.layers = args.layers
        self.embedding_dim = args.dim
        
        self.fc_in = nn.Linear(12, self.embedding_dim)
        self.fc_layers = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.layers)])
        self.fc_out = nn.Linear(self.embedding_dim, 6)

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for fc in self.fc_layers:
            x = torch.relu(fc(x))
        x = self.fc_out(x)
        return x
