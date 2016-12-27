import NeuralNetwork as nn
import PreprocessData as pp

import numpy as np
import random
import matplotlib.pyplot as plt
import math
from operator import add
import time

# This file is used for testing the neural network
def crossValidate(net, nb_folds, iterations=1000, learning_rate=0.01, grad_decay=0.9, epsilon=0.000001, adadelta=False):
    # Splits the data into nb_folds batches using each batch as a testing set in turn and rest as the training set

    ######## Need to fix: how to train on multiple years at once?
    data_trains, data_tests = pp.preprocessing_cross_valid(2012, 2014, nb_folds)
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
        start = time.clock()

        net.reset()
        # Make test and training sets
        x_train = data_trains[i][:, 1:]
        y_train = data_trains[i][:, 0]

        x_test = data_tests[i][:, 1:]
        y_test = data_tests[i][:, 0]

        temp = net.test(x_train, y_train, iterations, learning_rate, grad_decay, epsilon, adadelta, X_test=x_test, y_test=y_test)

        min_errs.append(temp[0])
        test_errs.append(temp[1])
        train_errs.append(temp[2])

        freqs = net.testProbBuckets(x_train, y_train, nb_buckets=nb_buckets, X_test=x_test, y_test=y_test)
        # Aggregates the prob buckets from each fold together
        freq_probs_test = list(map(add, freq_probs_test, freqs[0]))
        freq_wins_test = list(map(add, freq_wins_test, freqs[1]))
        freq_probs_train = list(map(add, freq_probs_train, freqs[2]))
        freq_wins_train = list(map(add, freq_wins_train, freqs[3]))


        print("Time:", time.clock() - start)

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


def makeOneFold(nb_folds):
    # Returns one fold from the cross-validation training set
    # Note: has to create the whole cross-validation set (could be improved)
    data_trains, data_tests = pp.preprocessing_cross_valid(2012, 2014, nb_folds)
    rand_fold = random.randint(0, nb_folds-1)  # Pick a random fold to test

    np.random.shuffle(data_trains[rand_fold])  # shuffles training examples

    x_train = data_trains[rand_fold][:, 1:]
    y_train = data_trains[rand_fold][:, 0]

    x_test = data_tests[rand_fold][:, 1:]
    y_test = data_tests[rand_fold][:, 0]

    return x_train, y_train, x_test, y_test

def testOneRun(net, nb_folds, iterations=1000, learning_rate=0.01, grad_decay=0.9, epsilon=0.000001, adadelta=False):
    # Takes one fold from the cross-validation set and tests it
    x_train, y_train, x_test, y_test = makeOneFold(nb_folds)

    # x_temp = x_train[:, 0:4]
    # x_temp2 = x_test[:, 0:4]
    #
    # # TEST remove wins, points, goals for - goals against, shots for - shots against
    # x_train = x_train[:, 4:]
    # x_test = x_test[:, 4:]
    # x_train = np.concatenate((x_train, x_temp), axis=1)
    # x_test = np.concatenate((x_test, x_temp2), axis=1)

    start = time.clock()
    temp = net.test(x_train, y_train, iterations, learning_rate, grad_decay, epsilon, adadelta, X_test=x_test, y_test=y_test)

    print("Time:", time.clock() - start)

    return temp[0]


def sequentialValidate(net, start=0.5, step=1, iterations=1000, learning_rate=0.01, grad_decay=0.9, epsilon=0.000001, adadelta=False):
    # Cross-validation procedure for time series data
    # Trains on the first 'start' fraction of examples and predicts the next one
    # Adds 'step' examples to training set and tests on the next example, repeat until all the examples have been used

    data = pp.preprocessing_final(2012, 2014, export=False)[0]
    x_data = data[:, 1:]
    y_data = data[:, 0]

    min_errs = []
    test_errs = []
    train_errs = []
    train_class_errs = []
    min_class_errs = []

    nb_examples = int(start * len(data))
    nb_runs = 0
    print(len(x_data[nb_examples]))
    while nb_examples < len(data):
        net.reset()
        temp = net.test(x_data[:nb_examples, :], y_data[:nb_examples], iterations, learning_rate, grad_decay, epsilon, adadelta, X_test=x_data[nb_examples:nb_examples+20, :], y_test=y_data[nb_examples:nb_examples+20])

        min_errs.append(temp[0])
        test_errs.append(temp[1])
        train_errs.append(temp[2])
        train_class_errs.append(temp[3])
        min_class_errs.append(temp[4])

        nb_examples += step
        nb_runs += 1

    print("\n----------")
    print(net, "\tNb runs:", nb_runs)
    print("Avg min:", sum(min_errs)/nb_runs, "\t\t\t", min_errs)
    print("Avg final test:", sum(test_errs)/nb_runs, "\t\t\t", test_errs)
    print("Avg final train:", sum(train_errs)/nb_runs, "\t\t\t", train_errs)
    print("Avg final class ", sum(train_class_errs)/nb_runs, "\t\t\t", train_class_errs)
    print("Avg min class ", sum(min_class_errs)/nb_runs, "\t\t\t", min_class_errs)

def hyperoptimization(iters):
    # Uses random search to find good hyperparameters
    # Number of hidden nodes per layer, weight decay, learning rate
    results = []

    start = time.clock()
    for i in range(iters):
        print("\n---- Optimization", i+1, "--")
        #s_time = time.clock()

        nb_hidden_nodes = int(random.uniform(30, 200)) #int(math.pow(10, random.uniform(1.5, 2.5)))
        weight_decay = math.pow(10, random.uniform(0, 1.5)) - 1 #math.pow(10, random.uniform(0, 1.5))
        learning_rate = math.pow(10, random.uniform(-2.5, -1.5)) #not relevant for adadelta
        grad_decay = 0.9
        epsilon = 0.0000001

        print(nb_hidden_nodes, weight_decay, learning_rate, "\n")

        net = nn.NeuralNetwork(34, nb_hidden_nodes, 1, nb_hidden_layers=1, weight_decay=weight_decay)
        min_err = testOneRun(net, 10, 500, learning_rate, grad_decay, epsilon)

        results.append((min_err, nb_hidden_nodes, weight_decay, learning_rate))

        #print("Time:", time.clock() - s_time)

    results.sort(key=lambda tup: tup[0])


    print("\n-- Total time: ", time.clock() - start)
    for i in range(len(results)):
        print(",".join(str(x) for x in results[i]))
    return


def trainingSizeTest(net, iterations, learning_rate, grad_decay=0.9, epsilon=0.000001, adadelta=False):
    # Plots error vs training set size for diagnosis
    x_train, y_train, x_test, y_test = makeOneFold(9)
    # for 10 folds, x_train is about 1000 examples now

    training_sizes = []
    min_errs = []
    train_errs = []

    for i in range(6, 21):
        net_clone = net.clone()
        # only train on a portion of examples
        nb_examples = i*100
        min_err, test_err, train_err, class_error, test_class_error = net_clone.test(x_train[:nb_examples], y_train[:nb_examples], iterations, learning_rate, grad_decay, epsilon, adadelta, X_test=x_test, y_test=y_test)

        training_sizes.append(nb_examples)
        min_errs.append(min_err)
        train_errs.append(train_err)

    plt.figure()
    plt.title('Error vs Nb Training Examples')
    plt.xlabel("Nb Training Examples")
    plt.ylabel('Error')

    plt.plot(training_sizes, train_errs, label="Training")

    plt.plot(training_sizes, min_errs, label="Test")
    plt.legend()

    return


if __name__ == '__main__':
    #random.seed(12)
    #np.random.seed(12)

    net = nn.NeuralNetwork(34, 120, 1, nb_hidden_layers=1, weight_decay=1)

    #trainingSizeTest(net, 500, 0.01)

    #net2 = net.clone()

    #testOneRun(net, 6, 300, learning_rate=0.0075, adadelta=False)
    #sequentialValidate(net, 0.75, 30, 500, 0.0055)

    #testOneRun(net2, 5, 500, adadelta=True)

    #crossValidate(net, 9, learning_rate=0.0075)
    hyperoptimization(20)

    #net.graphCosts()
    #net.graphWeights(False)
    #net2.graphCosts(1)
    #net2.graphWeights()
    plt.show()


