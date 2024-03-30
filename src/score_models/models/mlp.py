import torch
import torch.nn as nn


class ConditionalBatchNorm1d(nn.Module):
    """Conditional Batch Normalization 1d Layer."""

    def __init__(self, L: int, num_features: int) -> None:
        """Constructor of the Conditional Batch Normalization 1d Layer.

        :param L: Number of batch normalization layers
        :param num_features: Number of features in the input tensor
        """
        super().__init__()
        self.L = L
        self.num_features = num_features
        self.bn = nn.ModuleList([nn.BatchNorm1d(self.num_features) for _ in range(self.L)])

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Forward pass of the Conditional Batch Normalization 1d Layer.
        Passes the input tensor through the i-th batch normalization layer.

        :return: Output tensor
        """
        return self.bn[i](x)


class MLPScoreModel(nn.Module):
    """MLP Score Model."""

    def __init__(self, input_size: int, hidden_size: int, L: int) -> None:
        """Constructor of the MLP Score Model.

        :param input_size: Number of features in the input tensor
        :param hidden_size: Number of features in the hidden layer
        :param L: Number of batch normalization layers
        """
        super().__init__()
        self.input_size = input_size
        self.L = L

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn1 = ConditionalBatchNorm1d(L, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.bn2 = ConditionalBatchNorm1d(L, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Forward pass of the MLP Score Model. Computes the
        forward pass of the model through the i-th batch
        normalization layers.

        :param x: Input tensor
        :param i: Index of the batch normalization layer
        :return: Output tensor
        """
        x = self.bn1(self.relu1(self.fc1(x)), i)
        x = self.bn2(self.relu2(self.fc2(x)), i)
        return self.fc3(x)
