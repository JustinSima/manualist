""" Basic RNN to debug untrained right hand models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class URHMModel(nn.Module):
    def __init__(self, device, **rnn_args):
        """ Model with single 1d convolutional layer and n recurrent layers.

        Args:
            device: Torch device.
            T (int): Sequence length.
            n_layers (int): Desired number of recurrent layers.
        """
        super(URHMModel, self).__init__()
        self.device = device
        self.rnn = nn.RNN(
            batch_first=True,
            **rnn_args
        )
        self.rnn.to(device=device)

    def forward(self, input_tensor):
        """ Forward pass."""
        x = input_tensor.reshape(-1, 120, 63)
        _, x = self.rnn(x)
        print(f'RNN Output Shape: {x.shape}')
        x = x[0, :] # Change for bi-directional.
        x = x.reshape(-1, 25)

        return F.softmax(x)

class URHM(BaseModel):
    def __init__(
        self,
        device,
        label_mapping,
        training_loop,
        **rnn_args
    ):
        super().__init__(
            device=device,
            model=URHMModel(device=device, **rnn_args),
            label_mapping=label_mapping,
            training_loop=training_loop
        )
