#!/usr/bin/env python3
import csv
import inspect
import os
import sys

import numpy as np

from Functions import MSE, getAllData
from NeutralNetwork import NeutralNetwork

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

input_list = []
target_list = []
input_file = open(sys.argv[1], "r+")
input_list = getAllData(input_file)
target_list = input_list
input_file.close()

# HeadLines
results = []
results.append(["Learning rate", "Momentum", "Ilosc iteracji", "MSE", "STD"])

for k in range(100):
    input_nodes = 4
    hidden_nodes = 2
    output_nodes = 4
    learningrate = np.random.uniform(0, 0.7)
    bias = 1  # on
    momentum = np.random.uniform(0.1, 0.9)
    nn = NeutralNetwork(input_nodes, hidden_nodes,
                        output_nodes, learningrate, bias, momentum)
    i = 0
    error_test = 10
    std = "???"
    current_tuple = [learningrate, momentum]
    while error_test > 5 * 10**-3:
        if i > 10**5:
            break
        for j in range(len(input_list)):
            nn.train(input_list[j], target_list[j])
        f = nn.query
        error_test = MSE(f, input_list, target_list)
        """if i%1000==0:
            print("ERROR: ",error_test)
            print("i:",i)
            print("momentum: ",momentum)"""
        i += 1
    current_tuple.append(i)
    current_tuple.append(error_test)
    current_tuple.append(std)
    results.append(current_tuple)
    print(current_tuple)

with open("zad2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)
