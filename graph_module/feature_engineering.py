import numpy as np
import torch

class FeatureEngineer:

    def build_graph(self, net):
        self.net = net

        # Node features
        x = self.generate_node_features()

        # Edge index (connections)
        edges = []
        for _, line in net.line.iterrows():
            from_bus = int(line.from_bus)
            to_bus = int(line.to_bus)

            edges.append([from_bus, to_bus])
            edges.append([to_bus, from_bus])  # undirected graph

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "edge_index": edge_index
        }

    def generate_node_features(self):
        features = []

        for bus in range(len(self.net.bus)):
            voltage = 1.0
            p = 0.0
            q = 0.0

            if bus < len(self.net.load):
                p = self.net.load.p_mw.iloc[bus]
                q = self.net.load.q_mvar.iloc[bus]

            features.append([voltage, p, q])

        return np.array(features)