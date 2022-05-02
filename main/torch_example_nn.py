import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

# pytorch tensorboard --logdir=main/runs
writer = SummaryWriter("runs/MNIST")
# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01
stop_value = 0.0001

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Add image to tensorboard
examples = iter(test_loader)
example_data, example_target = examples.next()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('MNIST_image', img_grid)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Add model to tensorboard
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))

n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += (predictions == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / batch_size, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / batch_size, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

        if loss < stop_value:
            print(f'Epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')
            break

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    l_labels = []
    l_preds = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        l_preds.append(class_predictions)
        l_labels.append(predictions)

    # ????
    l_preds = torch.cat([torch.stack(batch) for batch in l_preds])
    l_labels = torch.cat(l_labels)
    classes = range(10)
    for i in classes:
        labels_i = l_labels == i
        preds_i = l_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy = {acc:.4f}')


FILE = "model.pth"
torch.save(model.state_dict(), FILE)
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(FILE, map_location=device))
model.to(device)


CHECK = "checkpoint.pth"
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}
torch.save(checkpoint, CHECK)

loaded_checkpoint = torch.load(CHECK, map_location=device)

epoch = loaded_checkpoint["epoch"]

model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(loaded_checkpoint["model_state"])
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(loaded_checkpoint["optim_state"])

