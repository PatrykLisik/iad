from functions import random_point, gaussRad, Euklides_dist


class RBF_neuron():
    def __init__(self, low=0, higth=10, input_number=1):
        self.c = random_point(input_number, low, higth)
        self.w = random_point(1, -4, 4)
        self.sig = random_point(1, -1, 1)

    def query(self, x):
        distance = Euklides_dist(self.c, x)
        return gaussRad(distance, self.sig) * self.w

    def train(self, input, output):
        pass

    def train_W(self):
        pass

    def train_sig(self):
        pass

    def train_C(self):
        pass
