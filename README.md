# earthquake_rnn


classification problem with nn.
using BCE instead of cross entropy since binary and not multiclass classification.
using 4 layers, with linear, ReLu, linear and Sigmoid at the end.
(Leaky ReLu > ReLu > TanH)
(SoftMax (CrossEntropyLoss) = multi-class. Sigmoid (BCELoss) = binary)
using SGD as optimizer.
doing batch training in n_epochs and n_iterations with batch_size = N.
list pre-processing done to data (scaling?).