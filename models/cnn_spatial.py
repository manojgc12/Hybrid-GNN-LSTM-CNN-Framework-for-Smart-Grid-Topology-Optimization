import torch
import torch.nn as nn
import numpy as np


class CNNSpatialExtractor(nn.Module):
    """
    1D CNN that slides across the bus voltage profile to detect
    spatial patterns such as voltage dips and overloaded zones.
    Input  : (batch=1, channels=1, num_buses)
    Output : feature vector of size output_dim
    """
    def __init__(self, num_buses=33, output_dim=16):
        super(CNNSpatialExtractor, self).__init__()
        self.conv1 = nn.Conv1d(1,  8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(16, output_dim)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.relu(self.fc(x))


def extract_spatial_features(net):
    """
    Run CNN over the voltage profile of the network.
    Returns: (feature_vector np.array, list of critical bus indices)
    """
    voltages  = net.res_bus.vm_pu.values.astype(np.float32)
    num_buses = len(voltages)

    model = CNNSpatialExtractor(num_buses=num_buses, output_dim=16)
    model.eval()

    v_tensor = torch.tensor(voltages).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        features = model(v_tensor)

    critical_buses = [int(i) for i, v in enumerate(voltages) if v < 0.97]
    return features.squeeze().numpy(), critical_buses


def report_spatial_analysis(net):
    """Print CNN spatial analysis and return features + critical buses."""
    features, critical_buses = extract_spatial_features(net)
    print("\nCNN Spatial Voltage Analysis:")
    print(f"  Feature vector shape          : {features.shape}")
    if critical_buses:
        print(f"  Voltage dip detected at buses : {critical_buses}")
        print(f"  Number of critical buses      : {len(critical_buses)}")
    else:
        print("  No critical voltage dip zones detected.")
    return features, critical_buses
