import numpy as np
import matplotlib.pyplot as plt
import math
#from sklearn.datasets import make_classification
import random

class NeuralNetwork:
    def __init__(self, nb_features, nb_nodes_per_layer, nb_outputs, nb_hidden_layers=1, weight_decay=0.5):

        self.weight_decay = weight_decay

        # For graphing
        self.train_error = []
        self.test_error = []
        self.iterations = []

        #np.random.seed(5)
        # Initialize weight matrices to random values
        # Each hidden layer gets its own matrix

        # Other hidden layer weights
        self.hid_weights = [np.random.uniform(-1, 1, (nb_nodes_per_layer, nb_nodes_per_layer + 1)) for i in range(nb_hidden_layers - 1)]

        # First hidden layer weights
        self.hid_weights.insert(0, np.random.uniform(-1, 1, (nb_nodes_per_layer, nb_features + 1)))

        # Output layer weights
        self.out_weights = np.random.uniform(-1, 1, (nb_outputs, nb_nodes_per_layer + 1))

        # dimensions of matrices: (number of nodes in the next layer, number of nodes in the current layer + 1)
        # add 1 to number of features for the bias unit

    def train(self, X, y, iterations=100, learning_rate=0.35, test_X=None, test_y=None, showCost=False):
        # X and y are your data
        # X is the set of features
        # y is the set of target values

        # Create copy of data for getCost (with no column of 1s)
        X2 = X

        # Add column of ones to X for the bias unit
        X = np.insert(X, 0, 1, 1)

        for j in range(iterations):
            # Populates the lists for cost graph
            if j % 5 == 0:
                self.train_error.append(self.getCost(X2, y))
                self.iterations.append(j)
                if showCost:
                    print("Cost:", self.train_error[-1])
                if test_X is not None and test_y is not None:
                    self.test_error.append(self.getCost(test_X, test_y))


            # TESTING decaying learning rate ###
            #if j % 100 == 0:
            #    learning_rate *= 0.9


            # Initialize sum of errors
            hid_deriv_sum = [np.zeros(matrix.shape) for matrix in self.hid_weights]
            out_deriv_sum = np.zeros(self.out_weights.shape)

            # Loop over data examples
            for i in range(len(X)):
                # Feed forward

                hid_activations = []
                # Hidden layer 1
                hid_activations.append(self.sigmoid(self.hid_weights[0].dot(X[i])))
                hid_activations[0] = np.insert(hid_activations[0], 0, 1)  # Add bias unit

                # Other hidden layers
                for k in range(1, len(self.hid_weights)):
                    hid_activations.append(self.sigmoid(self.hid_weights[k].dot(hid_activations[-1])))
                    hid_activations[-1] = np.insert(hid_activations[-1], 0, 1)

                # Output layer (sigmoid activation)
                out_activations = self.sigmoid2(self.out_weights.dot(hid_activations[-1]))

                # Backpropagation
                # Output errors (deltas)
                out_errors = out_activations - y[i]


                # Hidden layer errors
                hid_errors = []
                # Hidden layer -1
                hid_errors.insert(0, np.dot(self.out_weights.T, out_errors) * self.sigmoidPrime(hid_activations[-1]))
                hid_errors[0] = hid_errors[0][1:]  # remove bias term

                # Other hidden layers
                for k in reversed(range(1, len(self.hid_weights))):
                    hid_errors.insert(0, np.dot(self.hid_weights[k].T, hid_errors[0]) * self.sigmoidPrime(hid_activations[k-1]))
                    hid_errors[0] = hid_errors[0][1:]  # remove bias term


                # Update sum of errors
                # Hidden layer 1
                hid_deriv_sum[0] += np.dot(np.atleast_2d(hid_errors[0]).T, np.atleast_2d(X[i]))

                # Other hidden layers
                for k in range(1, len(self.hid_weights)):
                    #print(hid_activations[k-1])
                    #print(hid_errors[k])
                    hid_deriv_sum[k] += np.dot(np.atleast_2d(hid_errors[k]).T, np.atleast_2d(hid_activations[k-1]))

                # Output layer
                out_deriv_sum += np.dot(np.atleast_2d(out_errors).T, np.atleast_2d(hid_activations[-1]))


                # in_deriv_sum += np.dot(np.atleast_2d(l1_errors_no_bias).T, np.atleast_2d(X[i]))
                # out_deriv_sum += np.dot(np.atleast_2d(out_errors).T, np.atleast_2d(in_activations))

            # Update weights (include weight decay)
            hid_decay = [self.weight_decay * np.insert(w[:, 1:], 0, 0, 1) for w in self.hid_weights]
            out_decay = self.weight_decay * np.insert(self.out_weights[:, 1:], 0, 0, 1)  # don't regularize bias weights

            # Hidden layers
            for k in range(len(self.hid_weights)):
                self.hid_weights[k] -= learning_rate / len(X) * (hid_deriv_sum[k] + hid_decay[k])
            # Output layer
            self.out_weights -= learning_rate / len(X) * (out_deriv_sum + out_decay)

        return

    def predict(self, example):
        # Takes a single feature vector and outputs neural net's prediction
        example = np.insert(example, 0, 1)

        hid_activations = []
        # Hidden layer 1
        hid_activations.append(self.sigmoid(self.hid_weights[0].dot(example)))
        hid_activations[0] = np.insert(hid_activations[0], 0, 1)  # Add bias unit

        # Other hidden layers
        for k in range(1, len(self.hid_weights)):
            hid_activations.append(self.sigmoid(self.hid_weights[k].dot(hid_activations[-1])))
            hid_activations[-1] = np.insert(hid_activations[-1], 0, 1)

        # Output layer (sigmoid activation)
        out_activations = self.sigmoid2(self.out_weights.dot(hid_activations[-1]))

        return out_activations

    def predict_mult(self, examples):
        # Takes a list of features and outputs the prediction of each of them
        return [self.predict(ex) for ex in examples]

    def getCost(self, X, y, reg=False):
        # Calculates the cost function on the given examples
        # Use reg=False when comparing training/test error across different models

        # Use cross entropy error
        total_error = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total_error += -math.log(pred) if y[i] == 1 else -math.log(1-pred)

        # weight decay
        if reg:
            for w in self.hid_weights:
                total_error += self.weight_decay / 2 * np.square(w[:, 1:]).sum()  # weight decay (don't count bias unit weights)
            total_error += self.weight_decay / 2 * np.square(self.out_weights[:, 1:]).sum()

        return total_error / len(X)

    def classError(self, X, y):
        # Finds the classification rate
        total_error = 0
        for i in range(len(X)):
            # Using 0.5 as the threshold for deciding win/lose
            pred = self.predict(X[i])
            if (pred < 0.5 and y[i] == 1) or (pred >= 0.5 and y[i] == 0):
                total_error += 1

        return total_error / len(X)

    def sigmoid(self, x):
        # Have to try 1.719tanh(0.666666x) also
        return np.tanh(x)

    def sigmoidPrime(self, a):
        # Assumes a has already had the sigmoid function applied to it
        return 1 - a**2

    def sigmoid2(self, x):
        # This sigmoid is used for the output of the network (between 0 and 1)
        return 1 / (1 + np.exp(-x))

    def graphBoundary(self, x_data, t_data, spacing, x_range=None, y_range=None):
        # Creates a scatterplot of the data with the neural net's decision boundaries
        if x_range is None or y_range is None:
            maxs = np.amax(x_data, axis=0)
            mins = np.amin(x_data, axis=0)
            x_range = [mins[0] - spacing, maxs[0] + spacing]
            y_range = [mins[1] - spacing, maxs[1] + spacing]

        xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], spacing), np.arange(y_range[0], y_range[1], spacing))

        preds = np.array(self.predict_mult(np.c_[xx.ravel(), yy.ravel()]))
        preds = preds.reshape(xx.shape)

        plt.figure()
        plt.title('Decision boundaries')
        graph = plt.contour(xx, yy, preds,  cmap=plt.cm.Paired, levels=[0.1, 0.5, 0.9])
        plt.clabel(graph, inline=True, fontsize=10)


        plt.scatter(x_data[:, 0], x_data[:, 1], c=t_data, cmap=plt.cm.Paired)

        return

    def graphCosts(self, start=0):
        # Displays graph of costs
        plt.figure()
        plt.title('Cost vs Iteration')
        plt.xlabel("Iteration")
        plt.ylabel('Cost')

        plt.plot(self.iterations[start:], self.train_error[start:], label="Training")
        if self.test_error:
            plt.plot(self.iterations[start:], self.test_error[start:], label="Test")
        plt.legend()
        return


    def test(self, X, y, iterations, learning_rate, test_frac):
        # Splits the data into a training set and a test set
        # Prints the final training error and testing error (and the minimum test error)
        n = int(len(X) * test_frac)
        # Could shuffle examples first

        self.train(X[:n], y[:n], iterations, learning_rate, X[n:], y[n:])

        print("Train:", self.train_error[-1])
        print("Test:", self.test_error[-1])
        minErr = min(self.test_error)
        minIter = self.iterations[self.test_error.index(minErr)]
        print("Min:", minErr, "at", minIter, "iters")

        print("Train (class):", self.classError(X[:n], y[:n]))
        print("Test (class):", self.classError(X[n:], y[n:]))

        return minErr, self.test_error[-1], self.train_error[-1]

    def __repr__(self):
        return "Hidden layer weights:\n" + str(self.hid_weights) + "\nOutput layer weights:\n" + str(self.out_weights)

def testRuns(n, x, y):
    # Runs n tests and finds the average errors
    min_errs =[]
    final_errs = []
    train_errs = []
    for i in range(n):
        net = NeuralNetwork(2, 8, 1, 4, weight_decay=1)
        temp = net.test(x, y, 500, 0.3, 0.2)

        min_errs.append(temp[0])
        final_errs.append(temp[1])
        train_errs.append(temp[2])

    print(min_errs)
    print("Avg min:", sum(min_errs)/n)
    print(final_errs)
    print("Avg final test:", sum(final_errs)/n)
    print(train_errs)
    print("Avg final train:", sum(train_errs)/n)

    return

if __name__ == "__main__":
    net = NeuralNetwork(96, 96, 1, weight_decay=10)

    x = []
    y = []

    input = np.genfromtxt(
    'InputData2014-15_Final.csv',           # file name
    delimiter=',',          # column delimiter
    dtype='float32',        # data type
    filling_values=0,       # fill missing values with 0
    )

    random.shuffle(input)
    x = input[:, 1:]
    y = input[:, 0]

    net.test(x, y, 2000, 0.4, 0.2)

    net.graphCosts(50)

    # minsErr = []
    # minsIter = []
    #
    # tests = [2*i for i in range(4, 20)]
    # for j in tests:
    #     print("--- nodes =", j)
    #     net = NeuralNetwork(8, j, 1, weight_decay=0.4)
    #
    #     temp = net.test(x, y, 300, 1.5, 0.8)
    #     minsErr.append(temp[0])
    #     minsIter.append(temp[1])
    #
    # print(minsErr)
    # print(minsIter)
    # plt.plot(tests, minsErr, label="Error")
    # plt.title("Error vs nodes")
    # plt.figure()
    # plt.plot(tests, minsIter, label="Iter")
    # plt.title("Iter vs nodes")
    plt.show()
