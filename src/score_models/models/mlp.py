import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, t_size: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )

        self.temp_proj = nn.Sequential(
            nn.BatchNorm1d(t_size),
            nn.ReLU(),
            nn.Linear(t_size, output_size)
        )

    def forward(self, x, temb):
        x = self.block1(x)
        x += self.temp_proj(temb)
        return self.block2(x)


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

        self.time_embed = nn.Sequential(
            nn.Embedding(num_embeddings=self.L, embedding_dim=input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.head = nn.Linear(input_size, hidden_size)
        self.block1 = MLPBlock(hidden_size, hidden_size, hidden_size)
        self.block2 = MLPBlock(hidden_size, hidden_size, hidden_size)
        self.tail = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP Score Model. Computes the
        forward pass of the model through the i-th batch
        normalization layers.

        :param x: Input tensor
        :param i: Index of the batch normalization layer
        :return: Output tensor
        """
        temb = self.time_embed(i)
        x = self.head(x)
        x = self.block1(x, temb)
        x = self.block2(x, temb)
        return self.tail(x)