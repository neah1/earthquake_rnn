# earthquake_rnn
# tensorboard --logdir=main/datasets/runs

Model saving.
FILE = "model.pth"
torch.save(model.state_dict(), FILE)
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load(FILE, map_location=device))

K-Fold. SVM (80).
continuous time-series (cross-validation is different), might contradict causality when shuffling.
instead, take all data, keep 20 percent for testing/validation, u can shuffle the training data
(don't use anything before testing set) (cross-validation as moving window, moving year by year)