import numpy as np

from Lin_layer import Lin_layer
from RBF_layer import RBF_layer


class RBF():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr,
                 c_range=[-3, 3], sig_range=[-5, 5]):
        self.rbf = RBF_layer(input_nodes, hidden_nodes,
                             lr, c_range, sig_range)
        self.out = Lin_layer(hidden_nodes, output_nodes, lr)
        self.h_nodes = hidden_nodes

    def query(self, input):
        rbf_out = self.rbf.query(input)
        return self.out.query(rbf_out)[0]

    def train(self, input, target):
        """
        Train both layers
        """
        rbf_out = self.rbf.query(input)
        net_out = self.out.query(rbf_out)

        # weigth from output layer
        out_weigths = self.out.weigths
        output_errors = target - net_out
        hidden_errors = np.dot(out_weigths.T, output_errors)

        self.out.train(rbf_out, net_out, output_errors)
        self.rbf.train(input, rbf_out, hidden_errors)

    def train_lin(self, input, target):
        """
        Train output layer
        """
        rbf_out = self.rbf.query(input)
        net_out = self.out.query(rbf_out)

        output_errors = target - net_out

        self.out.train(rbf_out, net_out, output_errors)

    def set_up_centers_from_vec(self, input):
        """
        Set up centers as random vectros from X
        input: matrix of dimensions n x input_number
        """
        X = np.array(input)
        rnd_idx = np.random.permutation(X.shape[0])[:self.h_nodes]
        centers = [X[i, :] for i in rnd_idx]
        self.rbf.set_up_centers(centers)
