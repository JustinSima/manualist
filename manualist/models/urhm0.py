""" Basic RNN to debug untrained right hand models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class URHMModel(nn.Module):
    def __init__(self, device, **lstm_args):
        """ Model with single 1d convolutional layer and n recurrent layers.

        Args:
            device: Torch device.
            **lstm_args (dict): Desired number of recurrent layers.
        """
        super(URHMModel, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(
            batch_first=True,
            **lstm_args
        )
        self.lstm.to(device=device)

    def forward(self, sequences, lengths):
        """ Forward pass."""
        batch_size, seq_len, _ = sequences.size()
        sequences_packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths, batch_first=True)
        x, _ = self.lstm(sequences_packed)

        x, x_lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        x = x.view(batch_size*seq_len, 25)

        adjusted_lengths = [(l-1)*batch_size + i for i,l in enumerate(x_lengths)]
        length_tensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
        x = x.index_select(0, length_tensor)
        x = x.view(batch_size,25)
        
        return x

class URHM(BaseModel):
    """ Assumes the data loaders return packed sequences."""
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
