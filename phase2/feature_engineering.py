import torch
import numpy as np


class FeatureEngineer:
    """
    Converts a pandapower network object into a graph dict
    suitable for the GNN model.
    Node features per bus: [voltage_pu, p_load_mw, q_load_mvar]
    """

    def build_graph(self, net):
        num_buses = len(net.bus)

        voltages = net.res_bus.vm_pu.values.astype(np.float32)

        p_load = np.zeros(num_buses, dtype=np.float32)
        q_load = np.zeros(num_buses, dtype=np.float32)

        for idx, row in net.load.iterrows():
            bus = int(row["bus"])
            if bus < num_buses:
                p_load[bus] += float(row["p_mw"])
                q_load[bus] += float(row["q_mvar"])

        node_features = np.column_stack([voltages, p_load, q_load])
        x = torch.tensor(node_features, dtype=torch.float32)

        edges = []
        for _, line in net.line.iterrows():
            if line["in_service"]:
                edges.append([int(line["from_bus"]), int(line["to_bus"])])
                edges.append([int(line["to_bus"]),   int(line["from_bus"])])

        if not edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                edges, dtype=torch.long
            ).t().contiguous()

        return {"x": x, "edge_index": edge_index}
