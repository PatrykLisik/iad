import numpy as np

from functions import Euklides_dist, gaussRad, random_point


class RBF_neuron():
    def __init__(self, low=-3, higth=3, input_number=1, lr=0.2):
        self.c = random_point(input_number, low, higth)
        self.w = random_point(1, -4, 4)
        self.sig = random_point(1, -0.5, 0.5)
        """self.c = 5
        self.w = 0.4
        self.sig = 0.22"""
        self.lr = lr

    def query(self, x):
        distance = Euklides_dist(self.c, x)
        return gaussRad(distance, self.sig) * self.w

    def train(self, input, output, error):
        pass
        self.train_W(error)
        self.train_sig(error, output, input)

    def train_W(self, error):
        self.w += (self.lr * error * self.w)

    def train_sig(self, error, output, input):
        distance = Euklides_dist(self.c, input)
        d = error * gaussRad(distance, self.sig) * \
            (1 / np.sqrt(2 * np.pi) + (2 * self.sig)**-3)
        self.sig += d

    def train_C(self):
        pass
