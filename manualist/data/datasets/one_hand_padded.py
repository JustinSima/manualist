""" Module for loading samples and dataset. 
Returns arrays with 120 frames, zero padded.
"""
import os
import json
import torch
import numpy as np

import data.datasets.data_source  as ds


""" Loading a single sample."""
def load_sample(file_path, label_embedding, time_steps=120):
    # Load data.
    with open(file_path, 'r') as f:
        sample_dict = json.load(f)

    # Check for sample_dict.
    if 'null' in sample_dict.keys():
        raise KeyError('SAMPLE NOT FOUND.')

    # Containers and constants.
    hand = []
    n_frames = len(sample_dict['features'])
    label = np.asarray(label_embedding[sample_dict['label']], dtype=np.float32)
    label = torch.from_numpy(label)

    # Iterate through frames and store values.
    frames = {int(k):v for k, v in sample_dict['features'].items()}
    for _, frame in sorted(frames.items()):

        # Convert first hand to array,
        if len(frame['hand0']) == 1:
            frame_hand = np.empty(shape=(3, 21))
        else:
            frame_hand = np.asarray([ pos for _, pos in frame['hand0'].items() ]).T

        # Append frame tensors to containers.
        hand.append(frame_hand)

    # Convert array shapes.
    hand = torch.tensor(np.asarray(hand), dtype=torch.float32)

    # Test.
    assert hand.shape == (n_frames, 3, 21)
    assert n_frames <= 120

    # Zero pad sequences on the right.
    if n_frames < 120:
        n_padding = time_steps - n_frames
        pad = torch.zeros(size=(n_padding, 3, 21))
        hand = torch.cat(tensors=(hand, pad), dim=0)

    # Test.
    assert hand.shape == (time_steps, 3, 21)

    return hand, label

""" Dataset."""
class OneHand120PaddedDataset(torch.utils.data.Dataset):
    def __init__(self, parent_dir, split, partitions, label_embedding):
        super().__init__()
        with open(partitions, 'r') as partition_file:
            self.partitions = json.load(partition_file)
        with open(label_embedding, 'r') as embedding_file:
            self.label_embedding = json.load(embedding_file)
        self.parent_dir = parent_dir
        self.split = split
        self.vid_ids = self.partitions[self.split]

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        file_path = os.path.join(self.parent_dir, self.split, str(self.vid_ids[idx]).zfill(5)+'.json')

        features, label = load_sample(file_path=file_path, label_embedding=self.label_embedding)

        return features, label

""" Subclass of DataSource."""
class OneHand120Padded(ds.DataSource):
    """ Purely a convenience wrapper for my preferences."""
    def __init__(self, directory, partitions, label_embedding, **kwargs):
        super().__init__()
        
        # Training dataset and loader.
        self.train_dataset = OneHand120Dataset(directory, 'train', partitions, label_embedding)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            **kwargs
        )

        # Validation dataset and loader.
        self.val_dataset = OneHand120Dataset(directory, 'val', partitions, label_embedding)
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            **kwargs
        )

        # Testing dataset and loader.
        kwargs.pop('batch_size')
        self.test_dataset = OneHand120Dataset(directory, 'test', partitions, label_embedding)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            **kwargs
        )
