import NeuralNetwork as nn
import numpy as np
import random
import matplotlib.pyplot as plt

# This file is used for testing the neural network

def testRuns(n, x, y):
    # Runs n tests and finds the average errors
    min_errs =[]
    final_errs = []
    train_errs = []
    for i in range(n):
        print("--- Run " + str(i+1) + " ---")
        net = nn.NeuralNetwork(96, 32, 1, 2, weight_decay=25)
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

    net = nn.NeuralNetwork(96, 32, 1, nb_hidden_layers=2, weight_decay=25)

    input = np.genfromtxt(
    'InputData2014-15_Final.csv',           # file name
    delimiter=',',          # column delimiter
    dtype='float32',        # data type
    filling_values=0,       # fill missing values with 0
    )

    random.seed()
    random.shuffle(input)
    x = input[:, 1:]
    y = input[:, 0]


    net.train(x, y, 200, 0.04, True)

    net.graphCosts()

    plt.show()

