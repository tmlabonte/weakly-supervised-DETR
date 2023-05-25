"""Defines simple multi-layer perceptron."""

# Imports PyTorch packages.
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Defines simple multi-layer perceptron."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initializes MLP with linear layers."""

        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        # Initializes desired number and dimensionality of linear layers.
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Applies MLP with ReLU activations."""

        for i, layer in enumerate(self.layers):
            # Performs ReLU activation on all except the last layer.
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

