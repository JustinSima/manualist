""" Old code; doesn't run anymore."""
import json

import numpy as np
import torch


def vec_to_word(array):
    print(type(array))
    array_squeezed = array.squeeze().float().to('cpu').round(decimals=4)
    print('Squeezed input:', array_squeezed)
    labels = []
    vectors = []

    with open('files/data/labels/label-embeddings-glove-twitter-25.json', 'r') as f:
        label_dict = json.load(f)

    for lab, vec in label_dict.items():
        labels.append(lab)
        vectors.append(torch.from_numpy(np.array(vec, dtype=np.float32)))

    for ind, vect in enumerate(vectors):
        assert array_squeezed.shape == vect.float().shape

        if torch.allclose(array_squeezed, vect.float().round(decimals=3), atol=0.01, rtol=0.0):
            return_label = labels[ind]

        if array_squeezed[0] == vect.float().round(decimals=4)[0]:
            print('HIT', vect)
        
        if abs(array_squeezed[0] - vect.float().round(decimals=4)[0]) < 0.001:
            print('RIGHT HERE', array_squeezed[0],vect.float().round(decimals=4)[0])

    return return_label
