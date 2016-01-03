import NeuralNetwork as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# This file is used for testing the neural network



def crossValidate(net, x, y, nb_folds):
    # Splits the data into nb_folds batches using each batch as a testing set in turn
    nb_per_fold = math.ceil(len(y) / nb_folds)  # round up so last batch is smaller

    min_test_errs = []
    test_errs = []
    train_errs = []

    for i in range(nb_folds):
        print("--- Fold " + str(i+1) + " ---")

        # Make test and training sets
        x_test = x[i*nb_per_fold:(i+1)*nb_per_fold]
        y_test = y[i*nb_per_fold:(i+1)*nb_per_fold]

        x_train = np.concatenate((x[:i*nb_per_fold], x[(i+1)*nb_per_fold:]), axis=0)
        y_train = np.concatenate((y[:i*nb_per_fold], y[(i+1)*nb_per_fold:]), axis=0)

        temp = net.test(x_train, y_train, 500, 0.25, X_test=x_test, y_test=y_test)

        min_test_errs.append(temp[0])
        test_errs.append(temp[1])
        train_errs.append(temp[2])

        net.testProbBuckets(x_train, y_train, X_test=x_test, y_test=y_test)

    print("\n----------")
    print(net)
    print(min_test_errs)
    print("Avg min:", sum(min_test_errs)/nb_folds)
    print(test_errs)
    print("Avg final test:", sum(test_errs)/nb_folds)
    print(train_errs)
    print("Avg final train:", sum(train_errs)/nb_folds)

    return

def testRuns(net, n, x, y):
    # Runs n tests and finds the average errors
    # Each run is
    min_errs =[]
    final_errs = []
    train_errs = []
    for i in range(n):
        print("--- Run " + str(i+1) + " ---")
        temp = net.test(x, y, 2000, 0.25, 0.3)

        min_errs.append(temp[0])
        final_errs.append(temp[1])
        train_errs.append(temp[2])
        net.testProbBuckets(x, y, 0.3)

    print(min_errs)
    print("Avg min:", sum(min_errs)/n)
    print(final_errs)
    print("Avg final test:", sum(final_errs)/n)
    print(train_errs)
    print("Avg final train:", sum(train_errs)/n)

    return

if __name__ == '__main__':

    input = np.genfromtxt(
    'InputData2014-15_Final.csv',           # file name
    delimiter=',',          # column delimiter
    dtype='float64',        # data type
    filling_values=0,       # fill missing values with 0
    )
    random.seed()
    random.shuffle(input)
    x = input[:, 1:]
    y = input[:, 0]

    net = nn.NeuralNetwork(96, 32, 1, nb_hidden_layers=2, weight_decay=25)
    crossValidate(net, x, y, 10)

    plt.show()

