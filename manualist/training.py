""" Functionalized training loop with optional link to WandB."""
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
# import wandb


def training_loop(
    project_name, experiment_config,
    epochs, model, optimizer, loss_fn,
    train_loader, val_loader,
    l1_norm=None, l2_norm=None,
    device=None
):
    """ Basic training loop.
    Commented lines log training performance to WandB.
    """
    # Start new Weights and Biases project.
    # wandb.init(project=project_name, entity='signai')
    # wandb.config = experiment_config

    # Set device for model and input tensors.
    if not device:
        device = (torch.device('mps') if torch.backends.mps.is_available()
                else torch.device('cuda') if torch.cuda.is_available()
                else torch.device.device('cpu'))

    # Make sure model is on the specified device.
    if not hasattr(model, 'device') or model.device != device:
        model.to(device=device)
    
    # Training loop.
    for epoch in tqdm(range(1, epochs+1), leave=True, desc='Epochs'):
        time_start = datetime.datetime.now()
        loss_train = 0.0

        for batch in tqdm(train_loader, leave=True, desc='Batches'):
            sequences = batch[0].to(device=device)
            sequence_lengths = batch[1]
            labels = batch[2].to(device=device)
            sequences_packed = nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths, batch_first=True)

            outputs = model(sequences_packed, sequence_lengths)
            print('OUTPUT shape', outputs.shape)
            loss = loss_fn(outputs, labels)
            print('LOSS succesfully calculated.')

            # Optional regularization if not included in optimizer.
            if l1_norm:
                l1_penalty = sum(param.pow(2).sum() for param in model.parameters())
                loss += l1_penalty

            if l2_norm:
                l2_penalty = sum(param.abs().sum() for param in model.parameters())
                loss += l2_penalty

            optimizer.zero_grad()
            print('PREPARING for backward step.')
            loss.backward()
            print('Backward done.')
            optimizer.step()
            print('Step done.')

            loss_train += loss.item()

        # Deactivate gradients, calculate validation loss.
        with torch.no_grad():
            loss_val = 0.0
            for batch_ in tqdm(val_loader, leave=True, desc='Validation'):
                val_features = batch_[0].to(device=device)
                val_lengths = batch_[1]
                val_labels = batch_[2].to(device=device)

                val_outputs = model(val_features, val_lengths)

                batch_loss = loss_fn(val_outputs, val_labels)
                loss_val += batch_loss.item()

        # Log metrics to Weights and Biases.
        # wandb.log({
        #     "loss": loss_train,
        #     "val_loss": loss_val
        # })
        # wandb.watch(model)

        # Prompt.
        time_elapsed = datetime.datetime.now() - time_start
        print(f'Epoch: {epoch}, \
            Average Time per Batch: {time_elapsed / len(train_loader)}, \
            Training Loss: {loss_train / len(train_loader)}, \
            Validation Loss: {loss_val / len(val_loader)}')
