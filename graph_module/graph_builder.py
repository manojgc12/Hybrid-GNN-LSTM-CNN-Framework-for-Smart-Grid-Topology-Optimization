import torch
from torch_geometric.data import Data

class GraphBuilder:
    def __init__(self, net, node_features):
        self.net = net
        self.node_features = node_features

    def build_graph(self):
        edge_index = []
        for _, line in self.net.line.iterrows():
            edge_index.append([line.from_bus, line.to_bus])
            edge_index.append([line.to_bus, line.from_bus])

        return Data(
            x=torch.tensor(self.node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t()
        )
