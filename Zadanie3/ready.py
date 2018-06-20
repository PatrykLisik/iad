import csv
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import norm, pinv


class RBF:

    def __init__(self, input_number, centers_num,           output_number):
        self.input_number = input_number
        self.output_number = output_number
        self.centers_num = centers_num
        self.centers = [np.random.uniform(-1, 1, input_number)
                        for i in range(centers_num)]
        self.sigma = 10
        self.W = np.random.random((self.centers_num, self.output_number))

    def _basisfunc(self, c, d):
        assert len(d) == self.input_number
        return np.exp(-self.sigma * norm(c - d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.centers_num), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x input_number
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.centers_num]
        self.centers = [X[i, :] for i in rnd_idx]

        # calculate activations of RBFs
        G = self._calcAct(X)

        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)

    def set_up_centers(self, input):
        """
        Set up centers as random vectros from X
        X: matrix of dimensions n x input_number
        """
        # choose random center vectors from training set
        X = np.array(input)
        rnd_idx = np.random.permutation(X.shape[0])[:self.centers_num]
        self.centers = [X[i, :] for i in rnd_idx]

    def query(self, X):
        """ X: matrix of dimensions n x input_number """

        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


def getData(intput):
    reader = csv.reader(intput)
    outX = []
    outY = []
    for row in reader:
        i = list(map(float, row[0].split(" ")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX, outY

    # train set
    train_input_list = []
    train_target_list = []

    intput_file = open(sys.argv[1], "r+")
    train_input_list, train_target_list = getData(intput_file)
    intput_file.close()

    # test set
    test_input_list = []
    test_target_list = []

    intput_file = open(sys.argv[2], "r+")
    test_input_list, test_target_list = getData(intput_file)
    intput_file.close()

    x = np.array(train_input_list)
    # set y and add random noise
    y = np.array(train_target_list)
    # y += random.normal(0, 0.1, y.shape)
    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.set_up_centers(x)
    rbf.train(x, y)

    # plot test data
    plt.figure(figsize=(12, 8))
    plt.scatter(test_input_list, test_target_list, color="blue")

    # plot learned model
    n = 1000
    x_all = x = np.mgrid[-5:5:complex(0, n)].reshape(n, 1)
    y_all = rbf.query(x_all)
    plt.plot(x_all, y_all, 'r-', linewidth=2)

    # plot rbfs
    plt.plot(rbf.centers, np.zeros(rbf.centers_num), 'gs')

    for c in rbf.centers:
        # RF prediction lines
        cx = np.arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    print(rbf.W.shape)
    plt.show()
