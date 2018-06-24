from functions import Euklides_dist
from RBF_neuron import RBF_neuron


class RBF_layer():
    def __init__(self, input_number, output_number, lr,
                 c_range, sig_range):
        self.rbf = [RBF_neuron(input_number, lr, c_range, sig_range)
                    for _ in range(output_number)]
        self.lr = lr
        centers = [n.c[0] for n in self.rbf]
        for n in self.rbf:
            centers.sort(key=lambda p: Euklides_dist(n.c, p))
            n.set_sig(centers[1])

    def query(self, input):
        return [neuron.query(input)for neuron in self.rbf]

    def query_one(self, input, rbf_index):
        assert rbf_index < len(self.rbf)
        return [self.rbf[rbf_index].query(input)]

    def train(self, input, outputs, output_errors):
        for neuron, output, error in zip(self.rbf, outputs, output_errors):
            neuron.train(input, output, error)
        for err in output_errors:
            self.bias_rbf += self.lr * err

    def set_up_centers(self, centers):
        assert len(centers) == len(self.rbf)
        for c, rbf in zip(centers, self.rbf):
            rbf.c = c
