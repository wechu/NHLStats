import NeuralNetwork as nn
import PreprocessData as pp

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from operator import add

# This file is used for testing the neural network

def crossValidate(net, nb_folds):
    # Splits the data into nb_folds batches using each batch as a testing set in turn and rest as the training set

    ######## Need to fix: how to train on multiple years at once?
    data_trains, data_tests = pp.preprocessing_cross_valid(2014, nb_folds)

    min_errs = []
    test_errs = []
    train_errs = []

    nb_buckets = 4  # Could make this a parameter
    freq_probs_test = [0] * nb_buckets
    freq_wins_test = [0] * nb_buckets
    freq_probs_train = [0] * nb_buckets
    freq_wins_train = [0] * nb_buckets

    for i in range(nb_folds):
        print("--- Fold " + str(i+1) + " ---")

        net.reset()
        # Make test and training sets
        x_train = data_trains[i][:, 1:]
        y_train = data_trains[i][:, 0]

        x_test = data_tests[i][:, 1:]
        y_test = data_tests[i][:, 0]

        temp = net.test(x_train, y_train, 1000, 0.30, X_test=x_test, y_test=y_test)

        min_errs.append(temp[0])
        test_errs.append(temp[1])
        train_errs.append(temp[2])

        freqs = net.testProbBuckets(x_train, y_train, nb_buckets=nb_buckets, X_test=x_test, y_test=y_test)
        # Aggregates the prob buckets from each fold together
        freq_probs_test = list(map(add, freq_probs_test, freqs[0]))
        freq_wins_test = list(map(add, freq_wins_test, freqs[1]))
        freq_probs_train = list(map(add, freq_probs_train, freqs[2]))
        freq_wins_train = list(map(add, freq_wins_train, freqs[3]))

    print("\n----------")
    print(net, "\tNb folds:", nb_folds)
    print("Avg min:", sum(min_errs)/nb_folds, "\t\t\t", min_errs)
    print("Avg final test:", sum(test_errs)/nb_folds, "\t\t\t", test_errs)
    print("Avg final train:", sum(train_errs)/nb_folds, "\t\t\t", train_errs)

    probs_test = [freq_wins_test[i]/ freq_probs_test[i] if freq_probs_test[i] != 0 else -1 for i in range(nb_buckets)]
    probs_train = [freq_wins_train[i]/ freq_probs_train[i] if freq_probs_train[i] != 0 else -1 for i in range(nb_buckets)]

    print("Total freq test:")
    print(freq_probs_test)
    print(freq_wins_test)
    print(["{0:.2f}".format(x) for x in probs_test])

    print("Total freq train:")
    print(freq_probs_train)
    print(freq_wins_train)
    print(["{0:.2f}".format(x) for x in probs_train])
    return

def testRuns(net, n, x, y):
    # Runs n tests and finds the average errors
    min_errs =[]
    test_errs = []
    train_errs = []

    for i in range(n):
        print("--- Run " + str(i+1) + " ---")
        net.reset()
        temp = net.test(x, y, 2000, 0.25, 0.3)

        min_errs.append(temp[0])
        test_errs.append(temp[1])
        train_errs.append(temp[2])
        net.testProbBuckets(x, y, 0.3)

    print("\n----------")
    print(net)
    print("Avg min:", sum(min_errs)/n, "\t\t\t", min_errs)
    print("Avg final test:", sum(test_errs)/n, "\t\t\t", test_errs)
    print("Avg final train:", sum(train_errs)/n, "\t\t\t", train_errs)

    return

if __name__ == '__main__':


    #random.seed(6)

    #np.random.seed(6)
    net = nn.NeuralNetwork(94, 72, 1, nb_hidden_layers=3, weight_decay=20)
    crossValidate(net, 2)

    #net.test(x, y, 1000, 0.3, 0.3)
    #net.graphCosts(5)

    plt.show()
    #
    # x = np.array([[1, 2, 3],
    #               [4, 5, 6]])
    # y = np.array([[0, 7, 8],
    #               [9, 9, 9]])
    #
    # z = [x, y]
    # print(z)
    # print(z[0])
    # print(z[0][:, 1:])

