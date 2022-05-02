import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Neural Network Pipeline
# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# 0.1) Pre-process data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) Design model (input size, output size, forward pass)
class NeuralNetBinary(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetBinary, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


n_samples, n_features = X.shape
model = NeuralNetBinary(input_size=n_features, hidden_size=5, output_size=1)
print(f'Samples = {n_samples}, Features = {n_features}')

# 2) Construct loss and optimizer
n_epochs = 1000
stop_value = 0.0001
learning_rate = 0.01

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_epochs):
    # Forward_pass: compute prediction
    y_pred = model(X_train)
    # Compute loss
    loss = criterion(y_pred, y_train)
    # Backward_pass: compute gradient
    loss.backward()
    # Update_weights
    optimizer.step()
    # Empty_grad
    optimizer.zero_grad()
    # Print intermediate results
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}: loss = {loss.item():.4f}')
    # Early stopping
    if loss < stop_value:
        print(f'Training stopped. Epoch {epoch + 1}: loss = {loss.item():.4f}')
        break

# 4) Plot results
with torch.no_grad():
    prediction = model(X_test).round()
    acc = prediction.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')

    # predicted = y_pred.detach()
    # plt.plot(X_test.numpy(), y_test.numpy(), 'ro')
    # plt.plot(X_test.numpy(), predicted.numpy(), 'b')
    # plt.show()
