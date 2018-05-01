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

from Functions import MSE, getAllData
from NeutralNetwork import NeutralNetwork


input_list = []
target_list = []
input_file = open(sys.argv[1], "r+")
input_list = getAllData(input_file)
target_list = input_list
input_file.close()

# HeadLines
results = []
results.append(["Learning rate", "Momentum", "Ilosc iteracji", "MSE", "STD"])

learning_rates = [0.01, 0.1, 0.2, 0.4, 0.7] + [0.1] * 5
momentums = [0] * 5 + list(np.linspace(0.0, 1, 5))

# Const params
input_nodes = 4
hidden_nodes = 2
output_nodes = 4
bias = 1  # on

for learningrate, momentum in zip(learning_rates, momentums):
    # Array with number of iteration when network reach given precision
    number_of_iters = []
    # Array with outputs from once full cycle
    current_tuple = []
    print("learningrate: ", learningrate)
    print("Momentum: ", momentum)
    for num in range(100):
        i = 0
        error_test = 10
        nn = NeutralNetwork(input_nodes, hidden_nodes,
                            output_nodes, learningrate, bias, momentum)
        while error_test > 5 * 10**-3:
            for j in range(len(input_list)):
                nn.train(input_list[j], target_list[j])
            f = nn.query
            error_test = MSE(f, input_list, target_list)
            i += 1

        number_of_iters.append(i)
        print("Number of iter: ", num)
    # Leaeraning rate
    current_tuple.append(learningrate)
    # Momentum
    current_tuple.append(momentum)
    # Mean of number of iter
    current_tuple.append(np.mean(number_of_iters))
    # Standard deviaon of  iter
    current_tuple.append(np.std(number_of_iters, ddof=1))
    results.append(current_tuple)
    print(current_tuple)

with open("zad2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
