import copy
import inspect
import os
import sys

import numpy as np

from Lin_layer import Lin_layer
from RBF_layer import RBF_layer

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


class RBF():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr,
                 c_range=[-3, 3], sig_range=[-5, 5]):
        self.rbf = RBF_layer(input_nodes, hidden_nodes,
                             lr, c_range, sig_range)
        self.out = Lin_layer(hidden_nodes, output_nodes, lr)
        self.h_nodes = hidden_nodes

    def query(self, input):
        rbf_out = self.rbf.query(input)
        return self.out.query(rbf_out)

    def query_one(self, input, neuron_index):
        return self.rbf.query_one(input, neuron_index)[0] * self.out.weigths[0][neuron_index]

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

    def set_up_centers_from_vec_rand(self, input):
        """
        Set up centers as random vectros from X
        input: matrix of dimensions n x input_number
        """
        X = np.array(input)
        rnd_idx = np.random.permutation(X.shape[0])[:self.h_nodes]
        centers = [X[i, :] for i in rnd_idx]
        self.rbf.set_up_centers(centers)

    def set_up_centers_from_vec(self, inp):
        """
        Set up centers as random vectros from X
        input: matrix of dimensions n x input_number
        """
        input = copy.deepcopy(inp)
        in_len = len(input)
        step = int(np.floor(in_len / self.h_nodes))
        step += 1
        input.sort()
        centers = input[::step]
        #print("in_len: ", in_len)
        #print("step: ", step)
        #print("len cetnters", len(centers))
        #print("self.h_nodes", self.h_nodes)
        #print("step: ", step)
        # print(centers)
        self.rbf.set_up_centers(centers)

    def set_up_centers_gas(self, input):
        from Zadanie2.SOM import Neuron_gas
        pass
