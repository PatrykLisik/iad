from RBF_layer import RBF_layer
from Lin_layer import Lin_layer
import numpy


class RBF():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr=0.001):
        self.rbf = RBF_layer(input_nodes, hidden_nodes, lr)
        self.out = Lin_layer(hidden_nodes, output_nodes, lr)

    def query(self, input):
        rbf_out = self.rbf.query(input)
        return self.out.query(rbf_out)[0]

    def train(self, input, target):
        rbf_out = self.rbf.query(input)
        net_out = self.out.query(rbf_out)

        # weigth from output layer
        out_weigths = self.out.weigths
        output_errors = target - net_out
        hidden_errors = numpy.dot(out_weigths.T, output_errors)

        self.out.train(rbf_out, net_out, output_errors)
        self.rbf.train(input, rbf_out, hidden_errors)
