import numpy as np
import matplotlib.pyplot as plt
import math
#from sklearn.datasets import make_classification
import random

class NeuralNetwork:
    def __init__(self, nb_features, nb_nodes_per_layer, nb_outputs, nb_hidden_layers=1, weight_decay=0.5):
        # Parameters
        self.weight_decay = weight_decay

        self.nb_features = nb_features
        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.nb_outputs = nb_outputs
        self.nb_hidden_layers = nb_hidden_layers

        # For graphing
        self.train_error = []
        self.test_error = []
        self.iterations = []

        # Initialize weight matrices to random values
        # Each hidden layer gets its own matrix
        b = 0.3  # uniform distribution bounds
        # Other hidden layer weights
        self.hid_weights = [np.random.uniform(-b, b, (self.nb_nodes_per_layer, self.nb_nodes_per_layer + 1)) for i in range(self.nb_hidden_layers - 1)]

        # First hidden layer weights
        self.hid_weights.insert(0, np.random.uniform(-b, b, (self.nb_nodes_per_layer, self.nb_features + 1)))

        # Output layer weights
        self.out_weights = np.random.uniform(-b, b, (self.nb_outputs, self.nb_nodes_per_layer + 1))

        # dimensions of matrices: (number of nodes in the next layer, number of nodes in the current layer + 1)
        # add 1 to number of features for the bias unit

    def reset(self):
        # Reinitializes all the parameters
        self.train_error = []
        self.test_error = []
        self.iterations = []

        self.hid_weights = [np.random.uniform(-1, 1, (self.nb_nodes_per_layer, self.nb_nodes_per_layer + 1)) for i in range(self.nb_hidden_layers - 1)]
        self.hid_weights.insert(0, np.random.uniform(-1, 1, (self.nb_nodes_per_layer, self.nb_features + 1)))
        self.out_weights = np.random.uniform(-1, 1, (self.nb_outputs, self.nb_nodes_per_layer + 1))

    def train(self, X, y, iterations=100, learning_rate=0.35, test_X=None, test_y=None, showCost=False):
        # X and y are your data
        # X is the set of features
        # y is the set of target values

        init_learning_rate = learning_rate  # save initial learning rate
        time_constant = 100

        # Create copy of data for getCost (with no column of 1s)
        X2 = X

        # Add column of ones to X for the bias unit
        X = np.insert(X, 0, 1, 1)

        # Used for Adadelta
        hid_sq_change = [np.zeros(matrix.shape) for matrix in self.hid_weights]
        out_sq_change = np.zeros(self.out_weights.shape)
        hid_sq_deriv = [np.zeros(matrix.shape) for matrix in self.hid_weights]
        out_sq_deriv = np.zeros(self.out_weights.shape)
        ada_decay = 0.95
        epsilon = 0.0000001
        hid_deriv = [np.zeros(matrix.shape) for matrix in self.hid_weights]
        out_deriv = np.zeros(self.out_weights.shape)
        hid_change = [np.zeros(matrix.shape) for matrix in self.hid_weights]
        out_change = np.zeros(self.out_weights.shape)

        # Minibatch update paramters
        minibatch_size = 100
        minibatch_nb = -1

        for j in range(iterations):
            # Populates the lists for cost graph
            if j % 5 == 0:
                self.train_error.append(self.getCost(X2, y))
                self.iterations.append(j)
                if showCost:
                    print("Cost:", self.train_error[-1])
                if test_X is not None and test_y is not None:
                    self.test_error.append(self.getCost(test_X, test_y))

            # Early stopping
            if j >= 100 and j % 50 == 0:
                stop = True
                for k in range(1, 6):
                    if self.test_error[-k-1] > self.test_error[-k]:
                        stop = False
                        break
                if stop:
                    break

            # Initialize sum of errors
            hid_deriv_batch = [np.zeros(matrix.shape) for matrix in self.hid_weights]
            out_deriv_batch = np.zeros(self.out_weights.shape)

            # Increment to next minibatch
            minibatch_nb += 1
            # Loop over minibatch examples
            for i in range(minibatch_nb*minibatch_size, (minibatch_nb+1)*minibatch_size):
                # Check for last example
                if i == len(X):
                    minibatch_nb = -1
                    break


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
                hid_deriv_batch[0] += np.dot(np.atleast_2d(hid_errors[0]).T, np.atleast_2d(X[i]))

                # Other hidden layers
                for k in range(1, len(self.hid_weights)):
                    hid_deriv_batch[k] += np.dot(np.atleast_2d(hid_errors[k]).T, np.atleast_2d(hid_activations[k-1]))

                # Output layer
                out_deriv_batch += np.dot(np.atleast_2d(out_errors).T, np.atleast_2d(hid_activations[-1]))

            # Compute weight decay
            hid_decay = [self.weight_decay * np.insert(w[:, 1:], 0, 0, 1) for w in self.hid_weights]
            out_decay = self.weight_decay * np.insert(self.out_weights[:, 1:], 0, 0, 1)  # don't regularize bias weights

            # Compute update using Adadelta
            # Note there is no learning rate
            # Compute gradient
            for k in range(len(self.hid_weights)):
                hid_deriv[k] = (hid_deriv_batch[k] + hid_decay[k]) / minibatch_size
            out_deriv = (out_deriv_batch + out_decay) / minibatch_size

            # Accumulate gradient
            for k in range(len(self.hid_weights)):
                hid_sq_deriv[k] = ada_decay * hid_sq_deriv[k] + (1-ada_decay) * np.square(hid_deriv[k])
            out_sq_deriv = ada_decay * out_sq_deriv + (1-ada_decay) * np.square(out_deriv)

            # Compute update
            for k in range(len(self.hid_weights)):
                hid_change[k] = - np.sqrt(hid_sq_change[k] + epsilon) / np.sqrt(hid_sq_deriv[k] + epsilon) * hid_deriv[k]
            out_change = - np.sqrt(out_sq_change + epsilon) / np.sqrt(out_sq_deriv + epsilon) * out_deriv


            # Accumulate updates
            for k in range(len(self.hid_weights)):
                hid_sq_change[k] = ada_decay * hid_sq_change[k] + (1-ada_decay) * np.square(hid_change[k])
            out_sq_change = ada_decay * out_sq_change + (1-ada_decay) * np.square(out_change)

            # Update weights
            for k in range(len(self.hid_weights)):
                self.hid_weights[k] += hid_change[k]
            self.out_weights += out_change

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

    def test(self, X, y, iterations, learning_rate, test_frac=0, X_test=None, y_test=None):
        # Trains the neural net and prints the final training and testing errors (and the minimum test error)
        # If X_test and y_test are given then those are used as the test sets
        # Or else test_frac is used to split the data into training and test sets

        n = int(len(X) * (1 - test_frac))

        if X_test is None and y_test is None:
            X_test = X[n:]
            y_test = y[n:]

        self.train(X[:n], y[:n], iterations, learning_rate, X_test, y_test)

        print("Train:", self.train_error[-1])
        print("Test:", self.test_error[-1])
        minErr = min(self.test_error)
        minIter = self.iterations[self.test_error.index(minErr)]
        print("Min:", minErr, "at", minIter, "iters")

        print("Train (class):", self.classError(X[:n], y[:n]))
        print("Test (class):", self.classError(X_test, y_test))

        return minErr, self.test_error[-1], self.train_error[-1]

    def testProbBuckets(self, X, y, nb_buckets=10, test_frac=0, X_test=None, y_test=None):
        # Test probability buckets
        # This assumes we have trained before

        n = int(len(X) * (1 - test_frac))

        if X_test is None and y_test is None:
            X_test = X[n:]
            y_test = y[n:]

        preds_train = self.predict_mult(X[:n])
        preds_test = self.predict_mult(X_test)

        freq_probs_test = [0] * nb_buckets
        freq_wins_test = [0] * nb_buckets

        for x in range(len(preds_test)):
            for i in range(nb_buckets):
                if preds_test[x] >= i / nb_buckets and preds_test[x] < (i+1) / nb_buckets:
                    freq_probs_test[i] += 1
                    freq_wins_test[i] += int(y_test[x])

        freq_probs_train = [0] * nb_buckets
        freq_wins_train = [0] * nb_buckets

        for x in range(len(preds_train)):
            for i in range(nb_buckets):
                if preds_train[x] >= i / nb_buckets and preds_train[x] < (i+1) / nb_buckets:
                    freq_probs_train[i] += 1
                    freq_wins_train[i] += int(y[x])

        probs_test = [freq_wins_test[i]/ freq_probs_test[i] if freq_probs_test[i] != 0 else -1 for i in range(nb_buckets)]
        probs_train = [freq_wins_train[i]/ freq_probs_train[i] if freq_probs_train[i] != 0 else -1 for i in range(nb_buckets)]

        print("Freq test:")
        print(freq_probs_test)
        print(freq_wins_test)
        print(["{0:.2f}".format(x) for x in probs_test])

        print("Freq train:")
        print(freq_probs_train)
        print(freq_wins_train)
        print(["{0:.2f}".format(x) for x in probs_train])

        return freq_probs_test, freq_wins_test, freq_probs_train, freq_wins_train

    def __repr__(self):
        return "Nb hidden layers: " + str(self.nb_hidden_layers) + "\tNb hidden nodes per layer: " + str(self.nb_nodes_per_layer) + \
               "\tWeight decay: {0:.3f}".format(self.weight_decay)

if __name__ == "__main__":

    pass
