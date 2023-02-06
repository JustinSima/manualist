""" Base model on which to attach experimental pytorch models.
Used to train, make predictions with, and save models.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel:
    def __init__(self, device, model=None, label_mapping=None, training_loop=None) -> None:
        self.model = model
        self.device = device

        with open(label_mapping, 'r') as f:
            self.label_mapping = json.load(f)
        self.label_embeddings = [val for val in self.label_mapping.values()]

        self.training_loop = training_loop

    def train(self, **kwargs):
        self.training_loop(model=self.model, **kwargs)

    def predict(self, input_tensor):
        print('The correct predict was called.')
        raw_output = self.model(input_tensor).squeeze()

        # Find closest mapping value.
        closeness = [F.cross_entropy(raw_output, torch.from_numpy(np.array(emb, dtype=np.float32)).to(device=self.device)).to('cpu') for emb in self.label_embeddings]
        closest_embedding_ind = np.argmin(closeness)
        closest_embedding = self.label_embeddings[closest_embedding_ind]
        matching_label = [lab for ind, lab in enumerate(self.label_embeddings) if self.label_embeddings[ind] == closest_embedding]

        # Find and return key corresponding to this value.
        return torch.from_numpy(np.array(matching_label[0], dtype=np.float32))

    def predict_top_n(self, input_tensor):
        pass

    def save(self, file_name=''):
        torch.save(self.model, file_name)
