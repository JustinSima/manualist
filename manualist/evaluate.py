""" Evaluate model performance on a given data loader."""
import torch
import torch.nn as nn

from data.encoding.label_mapping import vec_to_word


def evaluate_performance(device, data_loader, model):
    count = 0
    count_correct = 0
    with torch.no_grad():
        for features, label in data_loader:
            features = features.to(device=device)
            label = label.to(device=device)
            pred = model.predict(features)

            predicted_label = vec_to_word(pred)
            actual_label = vec_to_word(label)

            count += 1
            if predicted_label == actual_label:
                count_correct += 1

    return count_correct / count
