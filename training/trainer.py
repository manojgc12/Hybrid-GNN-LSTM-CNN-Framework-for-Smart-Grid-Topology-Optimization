import torch
import torch.nn as nn
import numpy as np


def train_gnn(model, dataset, feature_engineer, epochs=25, lr=0.001):
    """
    Train GNN on multi-scenario dataset with normalized targets.
    Returns: loss_history, mean_loss, std_loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    loss_values = np.array([loss for _, loss in dataset], dtype=np.float32)
    mean_loss   = float(loss_values.mean())
    std_loss    = float(loss_values.std()) + 1e-8

    history = []

    print("\nGNN Multi-Scenario Training:")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for net_sample, true_loss in dataset:
            graph = feature_engineer.build_graph(net_sample)
            optimizer.zero_grad()

            pred       = model(graph)
            norm_tgt   = (true_loss - mean_loss) / std_loss
            target     = torch.tensor([[norm_tgt]], dtype=torch.float32)

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(dataset)
        history.append(avg)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Avg Loss: {avg:.8f}")

    return history, mean_loss, std_loss


def evaluate_gnn(model, dataset, feature_engineer, mean_loss, std_loss):
    """
    Evaluate trained GNN. Returns MSE and RMSE in original MW units.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for net_sample, true_loss in dataset:
            graph      = feature_engineer.build_graph(net_sample)
            pred_norm  = model(graph).item()
            pred       = pred_norm * std_loss + mean_loss
            errors.append((pred - true_loss) ** 2)

    mse  = float(np.mean(errors))
    rmse = float(np.sqrt(mse))
    return mse, rmse
