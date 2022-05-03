import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime

# FILE = "model.pth"
# torch.save(model.state_dict(), FILE)
# model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# model.load_state_dict(torch.load(FILE, map_location=device))

# Device and tensorboard # pytorch tensorboard --logdir=example/runs
writer = SummaryWriter("./runs/" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 28 * 28
hidden_size = 100
output_size = 10
n_epochs = 1
batch_size = 100
learning_rate = 0.01
stop_value = 0.0001

# 0) Prepare data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(test_loader)
example_data, example_target = examples.next()
img_grid = torchvision.utils.make_grid(example_data)


# 1) Design model (input size, output size, forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, output_size).to(device)
writer.add_graph(model, example_data.reshape(-1, 28 * 28).to(device))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

running_loss = 0.0
running_samples = 0
running_correct = 0
n_total_steps = len(train_loader)

for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item()
        running_samples += labels.shape[0]
        running_correct += (predictions == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / running_samples, epoch * n_total_steps + i)
            running_loss = 0.0
            running_samples = 0
            running_correct = 0

        if loss < stop_value:
            print(f'Epoch {epoch + 1}/{n_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            break

with torch.no_grad():
    samples = 0
    correct = 0
    all_labels = []
    all_predictions = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        samples += labels.shape[0]
        correct += (predictions == labels).sum().item()

        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        all_labels.append(labels)
        all_predictions.append(class_predictions)

    # Write PR-Curve
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat([torch.stack(batch) for batch in all_predictions])
    classes = range(10)
    for i in classes:
        label = all_labels == i
        prediction = all_predictions[:, i]
        writer.add_pr_curve(str(i), label, prediction, global_step=0)

    accuracy = correct / samples
    writer.add_text('Accuracy', str(accuracy))
    print(f'Accuracy = {accuracy:.4f}')
