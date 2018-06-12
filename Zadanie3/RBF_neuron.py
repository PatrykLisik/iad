import numpy as np

from functions import Euklides_dist, gaussRad, random_point


class RBF_neuron():
    def __init__(self, input_number, lr,  c_range,
                 sig_range):
        self.c = random_point(input_number, c_range[0], c_range[1])
        self.sig = random_point(1, sig_range[0], sig_range[1])
        self.lr = lr

    def query(self, x):
        distance = Euklides_dist(self.c, x)
        return gaussRad(distance, self.sig)

    def train(self, input, output, error):
        self.train_W(error)
        self.train_sig(error, output, input)

    def train_W(self, error):
        # self.w += (self.lr * error * self.w)
        pass

    def train_sig(self, error, output, input):
        distance = Euklides_dist(self.c, input)
        d = error * gaussRad(distance, self.sig) * \
            (1 / np.sqrt(2 * np.pi) + (2 * self.sig)**-3)
        self.sig += d

    def train_C(self):
        pass
