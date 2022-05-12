import numpy as np
import pandas as pd
import sys
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import datetime

# Device and tensorboard # tensorboard --logdir=main/runs
writer = SummaryWriter("./runs/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_size = 1
hidden_size = 2
num_classes = 1
num_layers = 1

# Training parameters
n_epochs = 1
batch_size = 100
learning_rate = 0.01
stop_value = 0.0001

# 0) Prepare data
sc = StandardScaler()
x_train = pd.read_pickle('./datasets/sets/x_train.pkl')
x_test = pd.read_pickle('./datasets/sets/x_test.pkl')
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
x_test = torch.from_numpy(x_test.astype(np.float32)).to(device)

y_train = pd.read_pickle('./datasets/sets/y_train.pkl')
y_test = pd.read_pickle('./datasets/sets/y_test.pkl')
y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Design model (input size, output size, forward pass)
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, num_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.sigm(out)
        return out


model = LSTM(num_classes, input_size, hidden_size, num_layers, x_train.shape[1]).to(device)
writer.add_graph(model, x_train[0].to(device))

# 2) Construct loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 3) Training loop
running_loss = 0.0
running_samples = 0
running_correct = 0
for epoch in range(n_epochs):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predictions = torch.max(outputs, 1)
    running_loss += loss.item()
    running_samples += y_train.shape[0]
    running_correct += (predictions == y_train).sum().item()

    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, loss = {loss.item():.4f}')
        writer.add_scalar('training loss', running_loss / 100, epoch)
        writer.add_scalar('accuracy', running_correct / running_samples, epoch)
        running_loss = 0.0
        running_samples = 0
        running_correct = 0

    if loss < stop_value:
        print(f'Halt. Epoch {epoch + 1}/{n_epochs}, loss = {loss.item():.4f}')
        break

# 4) Save results
with torch.no_grad():
    samples = 0
    correct = 0

    outputs = model(x_test)

    _, predictions = torch.max(outputs, 1)
    samples += y_test.shape[0]
    correct += (predictions == y_test).sum().item()

    # Write PR-Curve
    writer.add_pr_curve('pr_curve', y_test, predictions)
    accuracy = correct / samples
    writer.add_text('Accuracy', str(accuracy))
    print(f'Accuracy = {accuracy:.4f}')
