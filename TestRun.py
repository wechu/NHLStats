import NeuralNetwork as nn
import csv
import matplotlib.pyplot as plt

X = []
with open('InputData2014-15_Final.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append([float(entry) for entry in row])

csvfile.close()


y = []
with open('GameData2014-15_Clean.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        y.append([float(entry) for entry in row])

csvfile.close()

for i in range(4):
    print(X[i], y[i])

net = nn.NeuralNetwork(5, 47, 2)

net.train(X, y, 200, 0.04, True)


print(net.predict(X[0], y[0]))
net.graphCosts()

plt.show()

