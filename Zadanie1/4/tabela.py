#!/usr/bin/env python3
import csv
import inspect
import os
import sys

import numpy as np

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Functions import MSE, getData, round
from NeutralNetwork import NeutralNetwork


# train set
train_input_list = []
train_target_list = []

intput_file = open(sys.argv[1], "r+")
train_input_list, train_target_list = getData(intput_file)
intput_file.close()

# test set
test_input_list = []
test_target_list = []

intput_file = open(sys.argv[2], "r+")
test_input_list, test_target_list = getData(intput_file)
intput_file.close()

# HeadLines
results = []
headline = ("Number of neuron", "Średni błąd - zbiór treningowy", "Średni błąd - zbiór testowy",
            "Średnie odchylenie standarowe błąd - zbiór treningowy", "Średnie odchylenie standarowe - zbiór testowy")
results.append(headline)
print(headline)


number_of_iter = 2 * 10**2
input_nodes = 1
output_nodes = 1
learningrate = 0.1
bias = 1  # on
momentum = 0  # off


def activation_function_output(x): return x


def dactivation_function_output(x): return 1


for hidden_nodes_n in range(1, 20, 3):
    hidden_nodes = hidden_nodes_n
    error_test_tab = []
    error_train_tab = []
    for loop in range(100):
        nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes,
                            learningrate, bias, momentum,
                            activation_function_output,
                            dactivation_function_output)
        i = 0
        while i < number_of_iter:
            for j in range(len(train_input_list)):
                nn.train(train_input_list[j], train_target_list[j])
            i += 1
        f = nn.query
        # Copmute errors
        error_train = MSE(f, train_input_list, train_target_list)
        error_test = MSE(f, test_input_list, test_target_list)
        # Add to list
        error_train_tab.append(error_train)
        error_test_tab.append(error_test)
        # print("loop",loop)

    # Means
    error_test_mean = np.mean(error_test_tab)
    error_train_mean = np.mean(error_train_tab)
    # Std by default is population std ddof=1 is sample std
    error_test_std = np.std(error_test_tab, ddof=1)
    error_train_std = np.std(error_train_tab, ddof=1)

    # Make tuple and append to result
    tuple_to_append = (hidden_nodes, error_train_mean,
                       error_test_mean, error_train_std, error_test_std)
    tuple_to_append = tuple(map(round, tuple_to_append))
    results.append(tuple_to_append)
    print(tuple_to_append)

with open("TabelaNa4plik-{0}.csv".format(str(sys.argv[1])), "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
