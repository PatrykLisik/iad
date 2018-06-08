from RBF_neuron import RBF_neuron
from functions import random_point


class RBF_layer():
    def __init__(self, input_number, output_number, lr=0.1):
        self.rbf = [RBF_neuron(input_number=input_number, lr=lr)
                    for _ in range(output_number)]
        # aka w0
        self.bias_rbf = random_point(1, -4, 4)
        #self.bias_rbf = 0
        self.lr = lr

    def query(self, input):
        ret = []
        for neuron in self.rbf:
            ret.append(self.bias_rbf + neuron.query(input))
        return ret

    def train(self, input, outputs, output_errors):
        for neuron, output, error in zip(self.rbf, outputs, output_errors):
            neuron.train(input, output, error)
        for err in output_errors:
            self.bias_rbf += self.lr * err
