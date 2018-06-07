from RBF_neuron import RBF_neuron
from functions import random_point


class RBF_layer():
    def __init__(self, input_number, output_number):
        self.rbf = [RBF_neuron(input_number=input_number)
                    for _ in range(output_number)]
        self.bias_rbf = random_point(1, -4, 4)

    def query(self, input):
        ret = []
        for neuron in self.rbf:
            ret.append(self.bias_rbf + neuron.query(input))
        return ret

    def train(input, target):
        pass
