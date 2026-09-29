import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNModel(nn.Module):
    def __init__(self, input_dim=3):
        super(GNNModel, self).__init__()
        self.conv1   = GCNConv(input_dim, 32)
        self.conv2   = GCNConv(32, 64)
        self.fc1     = nn.Linear(64, 32)
        self.fc2     = nn.Linear(32, 1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index = data["x"], data["edge_index"]

        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))

        batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)

        x = self.relu(self.fc1(x))
        return self.fc2(x)
