import numpy as np
import matplotlib.pyplot as plt
import math

class NeuralNetwork:
    def __init__(self, nb_nodes_per_layer, nb_features, nb_outputs):

        self.nb_nodes_per_layer = nb_nodes_per_layer
        self.nb_features = nb_features
        self.nb_outputs = nb_outputs

        # For graphing
        self.costs = []
        self.iterations = []

        #np.random.seed(5)
        # Initialize weight matrices to random values
        # Each hidden layer gets its own matrix (only 1 hidden layer for now)
        self.l0_weights = np.random.uniform(-1, 1, (nb_nodes_per_layer, nb_features + 1))
        # Output layer weights
        self.l1_weights = np.random.uniform(-1, 1, (nb_outputs, nb_nodes_per_layer + 1))

        # dimensions of matrices: (number of nodes in the next layer, number of nodes in the current layer + 1)
        # add 1 to number of features for the bias unit

    def train(self, X, y, iterations=100, learning_rate=0.15, showCost=False):
        # X and y are your data
        # X is the set of features
        # y is the set of target values

        # Add column of ones to X for the bias unit
        X = np.insert(X, 0, 1, 1)

        for j in range(iterations):
            # Populates the lists for cost graph
            if j % 5 == 0:
                self.costs.append(self.getCost(X, y, False))
                self.iterations.append(j)

            # Print cost function
            if showCost and j % 10 == 0:
                print("Cost:", self.getCost(X, y, False))

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

            # Update weights
            self.l0_weights -= learning_rate * l0_deriv_sum / len(X)
            self.l1_weights -= learning_rate * l1_deriv_sum / len(X)

        return

    def predict(self, example, addOnes=True):
        # Takes a single feature vector and outputs neural net's prediction
        if addOnes:
            example = np.insert(example, 0, 1)

        l1_activations = self.sigmoid(self.l0_weights.dot(example))
        l1_activations = np.insert(l1_activations, 0, 1, 0)

        out_activations = self.sigmoid2(self.l1_weights.dot(l1_activations))
        return out_activations

    def predict_mult(self, examples):
        return [self.predict(ex) for ex in examples]


    def getCost(self, X, y, addOnes=True):
        # Use mean squared error
        total_error = 0
        for i in range(len(X)):
            total_error += ((self.predict(X[i], addOnes) - y[i])**2)/2
        # Could add a regularization term too (would need to change train())
        return total_error

    def sigmoid(self, x):
        # Have to try 1.719tanh(0.666666x) also
        return np.tanh(x)

    def sigmoidPrime(self, a):
        # Assumes a has already had the sigmoid function applied to it
        return 1 - a**2

    def sigmoid2(self, x):
        # This sigmoid is used for the output of the network (between 0 and 1)
        return 1 / (1 + np.exp(-x))

    def graphPredictions(self, X, y, X2=None):
        # Used for 1 feature data with 1 feature output (Eg: sin(x) = y)
        # X2 is used to check x-values other than those in the training set
        if X2 is None:
            X2 = X

        predictions = []
        for i in range(len(X2)):
            predictions.append(self.predict(X2[i]))

        plt.figure()
        plt.title('Comparing predictions and target')
        plt.xlabel('x-value')
        plt.ylabel('y-value')
        plt.scatter(X2, predictions, c='b', marker='.', label='NN Predictions')
        plt.scatter(X, y, c='g', marker='x', label='Target values')
        #plt.plot(X, y, color='g')
        plt.legend(bbox_to_anchor=(0.75, 0.95), loc=2, borderaxespad=0)
        return


    def graphBoundary(self, x_range, y_range, spacing):
        xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1], spacing), np.arange(y_range[0], y_range[1], spacing))

        preds = np.array(self.predict_mult(np.c_[xx.ravel(), yy.ravel()]))
        preds = preds.reshape(xx.shape)

        plt.figure()
        plt.title('Decision boundaries')
        graph = plt.contour(xx, yy, preds,  cmap=plt.cm.Paired, levels=[0.1, 0.5, 0.9])
        plt.clabel(graph, inline=True, fontsize=10)

        return

    def graphCosts(self):
        # Displays graph of costs (need to run train() with showCost=True first)
        plt.figure()
        plt.title('Cost vs Iteration')
        plt.xlabel("Iteration")
        plt.ylabel('Cost')
        plt.plot(self.iterations, self.costs)
        return

    def __repr__(self):
        return "Layer 0 weights:\n" + str(self.l0_weights) + "\nLayer 1 weights:\n" + str(self.l1_weights)

if __name__ == "__main__":

    # Tried out fitting a sine curve
    net = NeuralNetwork(8, 2, 1)

    # Test classification on XOR
    x = [[1, 0],
         [0, 1],
         [0, 0],
         [1, 1]]

    y = [1, 1, 0, 0]

    net.train(x, y, 200, 0.15, True)
    net.graphBoundary([-0.5, 1.5], [-0.5, 1.5], 0.05)
    net.train(x, y, 500, 0.15, True)
    net.graphBoundary([-0.1, 1.1], [-0.1, 1.1], 0.05)

    plt.show()