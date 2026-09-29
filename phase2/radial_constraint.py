import networkx as nx

def is_radial(net):

    G = nx.Graph()

    for _, line in net.line.iterrows():
        if line.in_service:
            G.add_edge(line.from_bus, line.to_bus)

    return nx.is_tree(G)