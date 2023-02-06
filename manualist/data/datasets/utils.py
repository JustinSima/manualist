""" Utilities that are useful for building datasets."""
import torch
import numpy as np


class PadSequence:
    """ Subsititute collation function when datasets are fed into RNNs.
    Assumes that each element is a tuple (features, labels).
    """
    def __call__(self, batch):
        # Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)

        # Store each sequences original length, then pad.
        sequences = [x[0] for x in sorted_batch]
        lengths = torch.from_numpy(np.array([len(x) for x in sequences], dtype=np.int64))
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # Store labels in order.
        labels = torch.stack([x[1] for x in sorted_batch])

        return sequences_padded, lengths, labels