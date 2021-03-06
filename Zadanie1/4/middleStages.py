#!/usr/bin/env python3
'''Arguments
-1 train_set
-2 test_set'''

import copy
import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from Functions import MSE, getData
from NeutralNetwork import NeutralNetwork

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def flat_list(d_list):
    return list(np.array(d_list).flat)


def plotChart(train_x, train_y, test_x, test_y, func_arry, title):
    # Set up plot
    plt.xlabel("x", fontsize="xx-large")
    plt.ylabel("y", fontsize="xx-large")
    train_x = flat_list(train_x)
    train_y = flat_list(train_y)
    test_x = flat_list(test_x)
    test_y = flat_list(test_y)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    # Plot train points
    plt.scatter(train_x, train_y, color='black',
                alpha=0.5, label="Punkty terningowe")
    # Plot test points
    plt.scatter(test_x, test_y, color='blue',
                alpha=0.1, label="Punkty testowe")
    # Plot given networks
    func_x = np.arange(min(train_x) - 1, max(train_x) + 1, 0.001)
    for iter, nn in func_arry.items():
        func_y = []
        for i in func_x:
            func_y.append(nn.query([i])[0][0])
        ax.plot(func_x, func_y, label="Epoki nauki {0}".format(iter))

    plt.grid()
    plt.title(title)
    # Legend
    plt.legend(loc='upper center',
               ncol=3, fancybox=True, shadow=True)
    plt.savefig(title + ".png")


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

# networks to plot
networks = {}
number_of_iteration = 600
iter_to_plot = np.linspace(0, number_of_iteration, 6)


input_nodes = 1
hidden_nodes = 5
output_nodes = 1
learningrate = 0.1
bias = 1
momentum = 0


def activation_function_output(x): return x


def dactivation_function_output(x): return 1


nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes,
                    learningrate, bias, momentum, activation_function_output,
                    dactivation_function_output)


i = 0
error_train2 = 5
error_train = 10
error_test = 10
while i < number_of_iteration:
    f = nn.query
    if(error_train2 == error_train):
        print("Learning end")
        break
    error_train2 = error_train
    error_train = MSE(f, train_input_list, train_target_list)
    error_test = MSE(f, test_input_list, test_target_list)
    if i % 1000 == 0:
        print("Iteracja: ", i)
        print("error_train: ", error_train)
        print("error_test: ", error_test)
        print("hidden_nodes: ", hidden_nodes)
    for j in range(len(train_input_list)):
        nn.train(train_input_list[j], train_target_list[j])
    if i in iter_to_plot:
        n = copy.deepcopy(nn)
        networks[i] = n
    i += 1


plotChart(train_input_list, train_target_list,
          test_input_list, test_target_list,
          networks,
          "Stany pośrednie: plik:{1}, Epoki nauki:{0} Neurony: {2}"
          .format(number_of_iteration, str(sys.argv[1]), hidden_nodes))
