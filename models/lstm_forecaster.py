import torch
import torch.nn as nn
import numpy as np


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def generate_load_sequence(num_hours=200, base_scale=1.0):
    """
    Synthetic 24-hour pattern load sequence.
    Morning peak ~8am, evening peak ~7pm.
    """
    np.random.seed(42)
    hours   = np.arange(num_hours)
    pattern = (
        0.6
        + 0.20 * np.sin(2 * np.pi * (hours % 24 - 8)  / 24)
        + 0.15 * np.sin(2 * np.pi * (hours % 24 - 19) / 24)
        + 0.05 * np.random.randn(num_hours)
    )
    return np.clip(pattern * base_scale, 0.5, 1.4)


def train_lstm(epochs=30, seq_len=24, forecast_steps=1):
    """
    Train LSTM on synthetic load sequence.
    Returns: model, mean_val, std_val, next_hour_forecast (scale factor)
    """
    load_data  = generate_load_sequence(num_hours=200)
    mean_val   = float(load_data.mean())
    std_val    = float(load_data.std()) + 1e-8
    normalized = (load_data - mean_val) / std_val

    X, y = [], []
    for i in range(len(normalized) - seq_len - forecast_steps):
        X.append(normalized[i : i + seq_len])
        y.append(normalized[i + seq_len : i + seq_len + forecast_steps])

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    model     = LSTMForecaster(input_dim=1, hidden_dim=64,
                               num_layers=2, output_dim=forecast_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn   = nn.MSELoss()

    print("\nLSTM Load Forecaster Training:")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        last_seq      = torch.tensor(
            normalized[-seq_len:], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1)
        forecast_norm = model(last_seq).item()
        forecast      = forecast_norm * std_val + mean_val

    final_loss = loss_fn(model(X).detach(), y).item()
    print(f"  LSTM RMSE                  : {float(np.sqrt(final_loss)):.6f}")
    print(f"  Predicted next-hour scale  : {forecast * 100:.1f}%")

    return model, mean_val, std_val, forecast
