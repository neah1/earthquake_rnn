from datetime import datetime
from math import ceil, floor

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from main.lstm_dataset import TimeSeriesDataset, DownSample, LSTM, device, LossCounter

# Training parameters
n_epochs = 100
n_freq = 3
batch_size = 50
learning_rate = 0.001
frequency = 100
T_length = 25
H_length = 3

# Model parameters
valid_size = 0.2
test_size = 0.2
shuffle = True
random_state = 42
hidden_size = 2
num_classes = 1
num_layers = 1

# 0) Prepare data
# TODO PR Curve. Parallel. K-FOLD (might be unnecessary). SVM (80). Over-fitting strategies.
dataset = TimeSeriesDataset('./datasets/sets/dataset.pkl', transform=DownSample(frequency, T_length, H_length))
x_i, idx_test, y_i, _ = train_test_split(range(len(dataset)), dataset.y, stratify=dataset.y, random_state=random_state,
                                         test_size=test_size)
idx_train, idx_valid, _, _ = train_test_split(x_i, y_i, stratify=y_i, random_state=random_state,
                                              test_size=valid_size / (1 - test_size))
train_split = Subset(dataset, idx_train)
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=shuffle)
valid_split = Subset(dataset, idx_valid)
valid_loader = DataLoader(valid_split, batch_size=batch_size, shuffle=shuffle)
test_split = Subset(dataset, idx_test)
test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=shuffle)

# 1) Create model, loss and optimizer
model = LSTM((T_length * frequency), hidden_size, num_classes, num_layers).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter("./runs/" + datetime.now().strftime("%d %b (%H-%M-%S)"))
writer.add_graph(model, iter(train_loader).next()[0])

# 2) Training loop
n_total_steps = len(train_loader)
n_steps = floor(n_total_steps / n_freq)
run_counter = LossCounter(n_steps)
train_counter = LossCounter(len(train_loader))
valid_counter = LossCounter(len(valid_loader))

for epoch in range(n_epochs):
    run_counter.reset()
    for i, (inp, labels) in enumerate(train_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_counter.update(loss.item(), labels, outputs)
        train_counter.update(loss.item(), labels, outputs)

        if (i + 1) % n_steps == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', run_counter.get_loss(), epoch * n_total_steps + i)
            writer.add_scalar('training accuracy', run_counter.get_acc(), epoch * n_total_steps + i)

    for i, (inp, labels) in enumerate(valid_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        valid_counter.update(loss.item(), labels, outputs)
    writer.add_scalars('validation loss',
                       {'train': train_counter.get_loss(), 'valid': valid_counter.get_loss()}, epoch)
    writer.add_scalars('validation accuracy',
                       {'train': train_counter.get_acc(), 'valid': valid_counter.get_acc()}, epoch)

# 3) Save results
with torch.no_grad():
    test_counter = LossCounter()
    for i, (inp, labels) in enumerate(test_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        test_counter.update(0, labels, outputs)
    labels, predictions = test_counter.get_results()
    writer.add_pr_curve('pr_curve', labels, predictions)
    writer.flush()

    accuracy = test_counter.get_acc()
    print(f'Accuracy = {accuracy:.4f}')
    params = f"EPOCHS: {n_epochs}, FREQ: {n_freq}, BATCH: {batch_size}, LR: {learning_rate}, HIDDEN: {hidden_size}, " \
             f"HZ: {frequency}, T: {T_length}, H: {H_length}, VALID: {valid_size}, TEST: {test_size}, " \
             f"SHUFFLE: {shuffle}, SEED: {random_state}, CLASSES: {num_classes}, LAYERS: {num_layers}, " \
             f"ACCURACY: {accuracy}. "
    writer.add_text('Parameters', str(params))
    writer.close()
