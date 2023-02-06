""" Untrained right hand models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class HandEmbedding(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n # The n-th step.
        self.hand_convolution = nn.Conv1d(
            in_channels=3, out_channels=1,
            kernel_size=1, stride=1
        )

    def forward(self, input_tensor):
        x = input_tensor[:, self.n, ::].reshape(shape=(-1, 3, 21))
        x = self.hand_convolution(x)

        return F.relu(x)


class URHMModel(nn.Module):
    def __init__(self, device, T: int, n_layers: int, output_dimension):
        """ Model with single 1d convolutional layer and n recurrent layers.

        Args:
            device: Torch device.
            T (int): Sequence length.
            n_layers (int): Desired number of recurrent layers.
        """
        super(URHMModel, self).__init__()
        self.device = device
        self.T = T
        self.embedding_layers = tuple(HandEmbedding(n=i) for i in range(0, 120))
        self.rnn = nn.RNN(
            input_size=21,
            hidden_size=output_dimension,
            batch_first=True,
            num_layers=n_layers
        )
        for layer in self.embedding_layers:
            layer.to(device=device)
        self.rnn.to(device=device)

    def forward(self, input_tensor):
        """ Forward pass."""
        embeddings = tuple(layer(input_tensor) for layer in self.embedding_layers)
        x = torch.cat(embeddings, dim=1)
        _, x = self.rnn(x)
        x = x.squeeze()

        return F.softmax(x)

class URHM(BaseModel):
    def __init__(
        self,
        device,
        label_mapping,
        training_loop,
        T,
        n_layers,
        output_dimension
    ):
        super().__init__(
            device=device,
            model=URHMModel(device=device, T=T, n_layers=n_layers, output_dimension=output_dimension),
            label_mapping=label_mapping,
            training_loop=training_loop
        )
