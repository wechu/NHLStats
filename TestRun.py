import NeuralNetwork as nn
import PreprocessData as pp

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from operator import add
import time

# This file is used for testing the neural network


def crossValidate(net, nb_folds, iterations=1000, learning_rate=0.4):
    # Splits the data into nb_folds batches using each batch as a testing set in turn and rest as the training set

    ######## Need to fix: how to train on multiple years at once?
    data_trains, data_tests = pp.preprocessing_cross_valid(2014, nb_folds)
    for i in range(nb_folds):
        np.random.shuffle(data_trains[i])  # shuffles training examples

    min_errs = []
    test_errs = []
    train_errs = []

    nb_buckets = 5  # Could make this a parameter
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

        temp = net.test(x_train, y_train, iterations, learning_rate, X_test=x_test, y_test=y_test)

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

    # Returns average min test error
    return sum(min_errs)/nb_folds

def testOneRun(net, nb_folds, iterations=1000, learning_rate=0.4):
    # Takes one fold from the cross-validation set and tests it
    data_trains, data_tests = pp.preprocessing_cross_valid(2014, nb_folds)
    rand_fold = random.randint(0, nb_folds-1)  # Pick a random fold to test

    np.random.shuffle(data_trains[rand_fold])  # shuffles training examples

    x_train = data_trains[rand_fold][:, 1:]
    y_train = data_trains[rand_fold][:, 0]

    x_test = data_tests[rand_fold][:, 1:]
    y_test = data_tests[rand_fold][:, 0]

    start = time.clock()
    temp = net.test(x_train, y_train, iterations, learning_rate, X_test=x_test, y_test=y_test)

    print("Time:", time.clock() - start)

    return temp[0]

def hyperoptimization(iters):
    # Uses random search to find good hyperparameters
    # Number of hidden nodes per layer, weight decay, learning rate
    results = []

    start = time.clock()
    for i in range(iters):
        print("\n---- Optimization", i+1, "--")
        #s_time = time.clock()

        nb_hidden_nodes = int(math.pow(10, random.uniform(1.5, 2.5)))
        weight_decay =  math.pow(10, random.uniform(0, 1.5))  #random.uniform(15, 25)
        learning_rate = 0  #random.uniform(0, 0.1) not relevant for adadelta

        print(nb_hidden_nodes, weight_decay, learning_rate, "\n")

        net = nn.NeuralNetwork(94, nb_hidden_nodes, 1, nb_hidden_layers=2, weight_decay=weight_decay)
        min_err = testOneRun(net, 5, learning_rate=learning_rate)

        results.append((min_err, nb_hidden_nodes, weight_decay, learning_rate))

        #print("Time:", time.clock() - s_time)

    results.sort(key=lambda tup: tup[0])


    print("\n-- Total time: ", time.clock() - start)
    for i in range(len(results)):
        print(",".join(str(x) for x in results[i]))
    return

if __name__ == '__main__':
    #random.seed(6)
    #np.random.seed(6)

    net = nn.NeuralNetwork(94, 100, 1, nb_hidden_layers=1, weight_decay=7)

    testOneRun(net, 5, 1000, 0)

    #crossValidate(net, 4, learning_rate=0.3)
    #hyperoptimization(10)

    net.graphCosts(1)

    plt.show()


