import torch
from power_system.network_loader import DistributionNetwork
from power_system.power_flow import PowerFlowAnalyzer
from graph_module.feature_engineering import FeatureEngineer
from graph_module.graph_builder import GraphBuilder
from models.gnn_model import GNNModel
from training.trainer import Trainer
from experiments.evaluation import compare_losses


network = DistributionNetwork()
net = network.load_network()
print("Network Summary:", network.summary())


pf = PowerFlowAnalyzer(net)
pf.run_power_flow()
initial_loss = pf.compute_total_loss()
print("Initial Power Loss:", initial_loss)

# Feature Engineering
features = FeatureEngineer().generate_node_features()

# Graph Construction
graph = GraphBuilder(net, features).build_graph()

# Model
model = GNNModel(input_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training
trainer = Trainer(model, optimizer, loss_fn)
target = torch.tensor([[initial_loss]], dtype=torch.float)
loss_history = trainer.train(graph, target)

# Evaluation
final_loss = loss_history[-1]
improvement = compare_losses(initial_loss, final_loss)

print("Final Training Loss:", final_loss)
print(f"Relative Improvement: {improvement:.2f}%")

# 