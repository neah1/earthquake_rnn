# earthquake_rnn
# tensorboard --logdir=main/datasets/runs

FILE = "model.pth"
torch.save(model.state_dict(), FILE)
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load(FILE, map_location=device))