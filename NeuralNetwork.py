import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_classification

class NeuralNetwork:
    def __init__(self, nb_features, nb_nodes_per_layer, nb_outputs, nb_hidden_layers=1, weight_decay=0.1):

        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.nb_features = nb_features
        self.nb_outputs = nb_outputs

        self.weight_decay = weight_decay

        # For graphing
        self.train_error = []
        self.test_error = []
        self.iterations = []

        #np.random.seed(5)
        # Initialize weight matrices to random values
        # Each hidden layer gets its own matrix (only 1 hidden layer for now)
        self.l0_weights = np.random.uniform(-1, 1, (nb_nodes_per_layer, nb_features + 1))
        # Output layer weights
        self.l1_weights = np.random.uniform(-1, 1, (nb_outputs, nb_nodes_per_layer + 1))

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
                    self.test_error.append(self.getCost(test_X, test_y, False))

            # TESTING decaying learning rate ###
            #if j % 100 == 0:
            #    learning_rate *= 0.9

            # Initialize sum of errors
            l0_deriv_sum = np.zeros(self.l0_weights.shape)  # Add 1 to count the bias unit's error
            l1_deriv_sum = np.zeros(self.l1_weights.shape)

            # Loop over data examples
            for i in range(len(X)):
                # Feed forward
                # Hidden layer 1
                l1_activations = self.sigmoid(self.l0_weights.dot(X[i]))
                l1_activations = np.insert(l1_activations, 0, 1)  # Add bias unit

                # Output layer (linear activation)
                l2_activations = self.sigmoid2(self.l1_weights.dot(l1_activations))

                # Backpropagation
                # Output errors (deltas)
                l2_errors = l2_activations - y[i]

                # Hidden layer 1 errors
                l1_errors = np.dot(self.l1_weights.T, l2_errors) * self.sigmoidPrime(l1_activations)

                # Update sum of errors
                # Remove the bias node from the errors
                l1_errors_no_bias = l1_errors[1:]

                l0_deriv_sum += np.dot(np.atleast_2d(l1_errors_no_bias).T, np.atleast_2d(X[i]))
                l1_deriv_sum += np.dot(np.atleast_2d(l2_errors).T, np.atleast_2d(l1_activations))

            # Update weights (include weight decay)
            l0_decay = self.weight_decay * np.insert(self.l0_weights[:, 1:], 0, 0, 1)  # don't regularize bias weights
            l1_decay = self.weight_decay * np.insert(self.l1_weights[:, 1:], 0, 0, 1)

            self.l0_weights -= learning_rate / len(X) * (l0_deriv_sum + l0_decay)
            self.l1_weights -= learning_rate / len(X) * (l1_deriv_sum + l1_decay)
        return

    def predict(self, example):
        # Takes a single feature vector and outputs neural net's prediction
        example = np.insert(example, 0, 1)

        l1_activations = self.sigmoid(self.l0_weights.dot(example))
        l1_activations = np.insert(l1_activations, 0, 1, 0)

        out_activations = self.sigmoid2(self.l1_weights.dot(l1_activations))
        return out_activations

    def predict_mult(self, examples):
        # Takes a list of features and outputs the prediction of each of them
        return [self.predict(ex) for ex in examples]

    def getCost(self, X, y, reg=True):
        # Calculates the cost function on the given examples
        # Use reg=False for test error since we don't count weight decay

        # Use cross entropy error
        total_error = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total_error += -math.log(pred) if y[i] == 1 else -math.log(1-pred)

        # weight decay
        if reg:
            total_error += self.weight_decay / 2 * np.square(self.l0_weights[:, 1:]).sum()  # weight decay (don't count bias unit weights)
            total_error += self.weight_decay / 2 * np.square(self.l1_weights[:, 1:]).sum()

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

        return minErr, minIter, self.test_error[-1]

    def __repr__(self):
        return "Layer 0 weights:\n" + str(self.l0_weights) + "\nLayer 1 weights:\n" + str(self.l1_weights)


if __name__ == "__main__":
    x, y = make_classification(n_samples=200, n_features=8, n_informative=2, n_redundant=2, n_classes=2, hypercube=True)

    net = NeuralNetwork(8, 16, 1, weight_decay=0.4)




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

def testRuns(n):
    # Runs n tests and finds the average minimum test error
    min_errs =[]
    errs = []
    for i in range(n):
        # Test classification on random data clusters
        x, y = make_classification(n_samples=200, n_features=8, n_informative=2, n_redundant=2, n_classes=2, hypercube=True)

        net = NeuralNetwork(8, 16, 1, weight_decay=0.4)
        temp = net.test(x, y, 200, 1.0, 0.8)
        min_errs.append(temp[0])
        errs.append(temp[2])

    print(min_errs)
    print("Avg min:", sum(min_errs)/n)
    print(errs)
    print("Avg final:", sum(errs)/n)

    return