from datetime import datetime
from math import ceil

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from main.lstm_dataset import TimeSeriesDataset, DownSample, LSTM, device

writer = SummaryWriter("./runs/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

# Training parameters
n_epochs = 100
n_steps = 10
batch_size = 50
test_size = 0.2
valid_size = 0.2
down_sample = 1
learning_rate = 0.001

# Model parameters
input_size = ceil(3001 / down_sample)
hidden_size = 5
num_classes = 1
num_layers = 1
shuffle = True
random_state = 42

# TODO shuffle (time-series), k-fold, sort based on time.
# TODO trim recording
# 0) Prepare data
dataset = TimeSeriesDataset(transform=DownSample(down_sample))
x_i, idx_test, y_i, _ = train_test_split(range(len(dataset)), dataset.y, stratify=dataset.y, random_state=random_state,
                                         test_size=test_size)
# TODO check split
idx_train, idx_valid, _, _ = train_test_split(x_i, y_i, stratify=y_i, random_state=random_state,
                                              test_size=valid_size / (1 - test_size))

train_split = Subset(dataset, idx_train)
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=shuffle)
valid_split = Subset(dataset, idx_valid)
valid_loader = DataLoader(valid_split, batch_size=batch_size, shuffle=shuffle)
test_split = Subset(dataset, idx_test)
test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=shuffle)

# 1) Create model, loss and optimizer
model = LSTM(input_size, hidden_size, num_classes, num_layers).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer.add_graph(model, iter(test_loader).next()[0])

# 2) Training loop
n_total_steps = len(train_loader)
for epoch in range(n_epochs):
    run_loss, run_correct, run_samples = (0.0, 0, 0)
    avg_loss, avg_correct, avg_samples = (0.0, 0, 0)
    avg_vloss, avg_vcorrect, avg_vsamples = (0.0, 0, 0)
    for i, (inp, labels) in enumerate(train_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = (torch.round(outputs) == labels).sum().item()

        avg_loss += loss.item()
        avg_correct += prediction
        avg_samples += labels.shape[0]

        run_loss += loss.item()
        run_correct += prediction
        run_samples += labels.shape[0]

        # TODO
        if (i + 1) % n_steps == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', run_loss / n_steps, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', run_correct / run_samples, epoch * n_total_steps + i)
            run_loss, run_correct, run_samples = (0.0, 0, 0)

    for i, (inp, labels) in enumerate(valid_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        avg_vloss += loss.item()
        avg_vcorrect += (torch.round(outputs) == labels).sum().item()
        avg_vsamples += labels.shape[0]

    avg_loss = avg_loss / len(train_loader) + 1
    avg_vloss = avg_vloss / len(valid_loader) + 1
    avg_acc = avg_correct / avg_samples
    avg_vcc = avg_vcorrect / avg_vsamples
    writer.add_scalars('validation loss', {'train': avg_loss, 'valid': avg_vloss}, epoch + 1)
    writer.add_scalars('validation accuracy', {'train': avg_acc, 'valid': avg_vcc}, epoch + 1)

# 3) Save results
with torch.no_grad():
    all_labels = []
    all_predictions = []
    for i, (inp, labels) in enumerate(test_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        all_labels.append(labels)
        all_predictions.append(torch.round(outputs))
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)
    accuracy = (all_predictions == all_labels).sum().item() / all_labels.shape[0]
    print(f'Accuracy = {accuracy:.4f}')

    writer.add_pr_curve('pr_curve', all_labels, all_predictions)
    params = f"TRAINING PARAMETERS: " \
             f"epochs: {n_epochs}, print_frequency: {n_steps}, batch: {batch_size}, lr: {learning_rate}, " \
             f"train: {1 - test_size}, valid: {valid_size}, test: {test_size}, HZ: {100 / down_sample}, " \
             f"seed: {random_state}, shuffle: {shuffle}. " \
             f"MODEL PARAMETERS: " \
             f"n_hidden: {hidden_size}, n_classes: {num_classes}, n_layers: {num_layers}. " \
             f"RESULTS: accuracy: {accuracy}. "
    writer.add_text('Parameters', str(params))
    writer.close()
