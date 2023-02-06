""" Load dataset and perform training loop."""
import torch

from data.datasets import OneHand120Padded
from models.urhm0 import URHM
from training import training_loop
from evaluate import evaluate_performance


# Training arguments.
batch_size = 32
epochs = 5
learning_rate = 1e-3

# Data loader.
one_hand_120 = OneHand120Padded(
    directory='files/data/',
    partitions='files/data/partitions-filtered-temp.json',
    label_embedding='files/data/labels/label-embeddings-glove-twitter-25.json',
    batch_size=batch_size
)

# Model.
device = torch.device('mps')
model = URHM(
    label_mapping='files/data/labels/label-embeddings-glove-twitter-25.json',
    training_loop=training_loop,
    device=device,

    # LSTM arguments.
    input_size=63,
    num_layers=1,
    hidden_size=25
)

# Optimizer and loss.
optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# Training loop.
model.train(
    project_name='Untrained Right Hand Only',
    experiment_config={
        'batch_size':batch_size,
        'epochs':epochs,
        'learning_rate':learning_rate
    },
    epochs=epochs,
    optimizer=optimizer,
    loss_fn=loss_function,
    train_loader=one_hand_120.train_loader,
    val_loader=one_hand_120.val_loader,
    device=device
)

# Evaluate.
accuracy = evaluate_performance(device=device, data_loader=one_hand_120.test_loader, model=model)
print(f'Final model accuracy: {accuracy}')
