from RBF_layer import RBF_layer
from Lin_layer import Lin_layer


class RBF():
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.rbf = RBF_layer(input_nodes, hidden_nodes)
        self.out = Lin_layer(hidden_nodes, output_nodes)

    def query(self, input):
        rbf_out = self.rbf.query(input)
        return self.out.query(rbf_out)

    def train(input, target):
        pass
