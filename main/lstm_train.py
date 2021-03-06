import sys
from math import floor
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from lstm_model import TimeSeriesDataset, DownSample, LSTM, device, LossCounter, EarlyStopper

# Training parameters
file = sys.argv[1]
versus = sys.argv[2] == 'True'
n_epochs = int(sys.argv[3])
patience = int(sys.argv[4])
learning_rate = float(sys.argv[5])
T_length = int(sys.argv[6])
HZ = int(sys.argv[7])

# Model parameters
H_length = 3
batch_size = 100
valid_size = 0.2
test_size = 0.2
random_state = 42

name = f'{file}_T{T_length}_H{H_length}_HZ{HZ}_E{n_epochs}_PT{patience}_LR{learning_rate}'
print(name)

# 0) Prepare data
dataset = TimeSeriesDataset(f'./datasets/{file}.pkl', transform=DownSample(HZ, T_length, H_length))
x_i, idx_test, y_i, _ = train_test_split(range(len(dataset)), dataset.y, stratify=dataset.y, shuffle=True,
                                         random_state=random_state, test_size=test_size)
idx_train, idx_valid, _, _ = train_test_split(x_i, y_i, stratify=y_i, shuffle=True,
                                              random_state=random_state, test_size=valid_size / (1 - test_size))
train_loader = DataLoader(Subset(dataset, idx_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(Subset(dataset, idx_valid), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(dataset, idx_test), batch_size=batch_size, shuffle=True)

# 1) Create model, loss and optimizer
model = LSTM(input_size=(T_length * HZ), hidden_size=2, num_classes=1, num_layers=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
writer = SummaryWriter("./datasets/runs/" + name)
writer.add_graph(model, iter(train_loader).next()[0])

# 2) Set training variables
last_epoch = n_epochs
n_total_steps = len(train_loader)
n_steps = floor(n_total_steps / 1)
early_stop = EarlyStopper(patience)
train_counter = LossCounter(len(train_loader))
valid_counter = LossCounter(len(valid_loader))

# 3) Model training and validation
for epoch in range(n_epochs):
    # Training loop
    for i, (inp, labels) in enumerate(train_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_counter.update(loss.item(), labels, outputs)
        if (i + 1) % n_steps == 0:
            print(f'Epoch: {epoch + 1}/{n_epochs}, step: {i + 1}/{n_total_steps}, loss: {loss.item():.4f}')
    # Validation loop
    for i, (inp, labels) in enumerate(valid_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        valid_counter.update(loss.item(), labels, outputs)
    # Plot loss and accuracy
    train_loss = train_counter.get_loss()
    valid_loss = valid_counter.get_loss()
    train_acc = train_counter.get_acc()
    valid_acc = valid_counter.get_acc()
    writer.add_scalar('Loss: Training', train_loss, epoch)
    writer.add_scalar('Loss: Validation', valid_loss, epoch)
    writer.add_scalar('Accuracy: Training', train_acc, epoch)
    writer.add_scalar('Accuracy: Validation', valid_acc, epoch)
    if versus:
        writer.add_scalars('Loss', {'training': train_loss, 'validation': valid_loss}, epoch)
        writer.add_scalars('Accuracy', {'training': train_acc, 'validation': valid_acc}, epoch)
    # Early stopping
    if early_stop.update(valid_loss):
        print('Early stopping')
        last_epoch = epoch + 1
        break

# 4) Save results
with torch.no_grad():
    test_counter = LossCounter()
    for i, (inp, labels) in enumerate(test_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        test_counter.update(0, labels, outputs)
    labels, predictions = test_counter.get_results()
    writer.add_pr_curve('PR Curve', labels, predictions)
    writer.flush()
    accuracy = test_counter.get_acc()
    print(f'Accuracy = {accuracy:.4f}')
    params = f"FILE: {file}, ACCURACY: {accuracy:.4f}, T: {T_length}, H: {H_length}, HZ: {HZ}, " \
             f"EPOCH: {last_epoch}/{n_epochs}, PATIENCE: {patience}, LR: {learning_rate}."
    writer.add_text('Parameters', str(params))
    writer.close()
