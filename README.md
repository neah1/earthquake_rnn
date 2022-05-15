# earthquake_rnn
classification problem with nn.
using BCE instead of cross entropy since binary and not multiclass classification.
using 4 layers, with linear, ReLu, linear and Sigmoid at the end.
(Leaky ReLu > ReLu > TanH)
(SoftMax (CrossEntropyLoss) = multi-class. Sigmoid (BCELoss) = binary)
using SGD as optimizer.
doing batch training in n_epochs and n_iterations with batch_size = N.
list pre-processing done to data (scaling?).

continuous time-series (cross-validation is different), might contradict causality when shuffling
instead take all data, keep 20 percent for testing/validation, u can shuffle the training data
(don't use anything before testing set). Don't contradict causality.
(cross-validation as moving window, moving year by year)

tensorboard --logdir=main/runs