import sys
from datetime import datetime
from math import floor
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from lstm_model import TimeSeriesDataset, DownSample, LSTM, device, LossCounter, EarlyStopper

# Training parameters
if len(sys.argv) > 1:
    T_length = int(sys.argv[1])
    H_length = int(sys.argv[2])
    HZ = int(sys.argv[3])
    n_epochs = int(sys.argv[4])
    patience = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
else:
    T_length = 30
    H_length = 3
    HZ = 50
    n_epochs = 100
    patience = 5
    learning_rate = 0.001

# Model parameters
batch_size = 100
valid_size = 0.2
test_size = 0.2
random_state = 42

# 0) Prepare data
# TODO PR-Curve. LSTM Parameters. Model saving. Over-fitting. K-FOLD. SVM (80).
print(f'Device: {device}, T: {T_length}, H: {H_length}, HZ: {HZ}, E: {n_epochs}, PTN: {patience}, LR: {learning_rate}')
dataset = TimeSeriesDataset('./datasets/sets/dataset.pkl', transform=DownSample(HZ, T_length, H_length))
x_i, idx_test, y_i, _ = train_test_split(range(len(dataset)), dataset.y, stratify=dataset.y, random_state=random_state,
                                         test_size=test_size)
idx_train, idx_valid, _, _ = train_test_split(x_i, y_i, stratify=y_i, random_state=random_state,
                                              test_size=valid_size / (1 - test_size))
train_split = Subset(dataset, idx_train)
train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
valid_split = Subset(dataset, idx_valid)
valid_loader = DataLoader(valid_split, batch_size=batch_size, shuffle=True)
test_split = Subset(dataset, idx_test)
test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=True)

# 1) Create model, loss and optimizer
model = LSTM(input_size=(T_length * HZ), hidden_size=2, num_classes=1, num_layers=1).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter("./datasets/runs/" + datetime.now().strftime("%d %b (%H-%M-%S)"))
writer.add_graph(model, iter(train_loader).next()[0])

# 2) Set training variables
last_epoch = n_epochs
n_total_steps = len(train_loader)
n_steps = floor(n_total_steps / 4)
early_stop = EarlyStopper(patience)
train_counter = LossCounter(len(train_loader))
valid_counter = LossCounter(len(valid_loader))

# 3) Training loop
for epoch in range(n_epochs):
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
    for i, (inp, labels) in enumerate(valid_loader):
        labels = labels.unsqueeze(1)
        outputs = model(inp)
        loss = criterion(outputs, labels)
        valid_counter.update(loss.item(), labels, outputs)
    train_loss = train_counter.get_loss()
    valid_loss = valid_counter.get_loss()
    train_acc = train_counter.get_acc()
    valid_acc = valid_counter.get_acc()
    writer.add_scalars('validation loss', {'train': train_loss, 'valid': valid_loss}, epoch)
    writer.add_scalars('validation accuracy', {'train': train_acc, 'valid': valid_acc}, epoch)
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
    writer.add_pr_curve('pr_curve', labels, predictions, )
    writer.flush()
    accuracy = test_counter.get_acc()
    print(f'Accuracy = {accuracy:.4f}')
    params = f"ACCURACY: {accuracy:.4f}, DEVICE: {device}, T: {T_length}, H: {H_length}, HZ: {HZ}, " \
             f"EPOCH: {last_epoch}/{n_epochs}, PATIENCE: {patience}, LR: {learning_rate}."
    writer.add_text('Parameters', str(params))
    writer.close()
