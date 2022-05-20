import numpy as np

# TODO HPC
# What strategies to try to avoid over-fitting, supposing it never goes away?
# pr curve, check where the errors are (false positive / negative are).
# select the better stations. station by station.
# check which channel to use for frequency.
# try using SVM (80).

# K-FOLD (might be unnecessary)
# continuous time-series (cross-validation is different), might contradict causality when shuffling.
# instead, take all data, keep 20 percent for testing/validation, u can shuffle the training data
# (don't use anything before testing set) (cross-validation as moving window, moving year by year)
