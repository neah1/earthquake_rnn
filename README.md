# earthquake_rnn
# tensorboard --logdir=main/runs

continuous time-series (cross-validation is different), might contradict causality when shuffling.
instead, take all data, keep 20 percent for testing/validation, u can shuffle the training data
(don't use anything before testing set) (cross-validation as moving window, moving year by year)