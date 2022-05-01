import torch
import torch.nn as nn

"""
empty, zeros, ones, rand, tensor
size(), view(), mean(), slice[:, :]
device(cuda, cpu), to(), numpy(), from_numpy()
requires_grad, w.detach(), with torch.no_grad():, x.grad.zero_()
"""


# Neural Network Pipeline

# 1) Design model (input size, output size, forward pass)

class NewModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NewModel, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

model = NewModel(input_size, output_size)

print(f'Samples = {n_samples}, Features = {n_features}')
print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')

# 2) Construct loss and optimizer
n_iters = 1000
stop_value = 0.00001
learning_rate = 0.01

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(n_iters):
    # Forward_pass: compute prediction
    y_pred = model(X)

    # Compute loss
    l = loss(Y, y_pred)

    # Backward_pass: compute gradient
    l.backward()

    # Update_weights
    optimizer.step()

    # Empty_grad
    optimizer.zero_grad()

    if epoch % 100 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch + 1}: weight = {w[0][0].item():.3f}, loss = {l:.8f}')

    if l < stop_value:
        print('Training stopped')
        break

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
