import copy
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

from RBF import RBF


def MSE(f, intput, output):
    ans = 0
    for i in range(len(output)):
        ans += ((f(intput[i]) - output[i])**2)
    return np.sum(ans) / len(output)


def getData(intput):
    reader = csv.reader(intput)
    outX = []
    outY = []
    for row in reader:
        i = list(map(float, row[0].split(" ")))
        outX.append(i[0:-1])
        outY.append(i[-1])
    return outX, outY


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


def flat_list(d_list):
    return list(np.array(d_list).flat)


def plotChart(train_x, train_y, test_x, test_y, func_arry, title):
    print("plot")
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
    colors = iter(cm.rainbow(np.linspace(0, 1, len(func_arry))))
    for neur, nn in func_arry.items():
        func_y = [nn.query([i]) for i in func_x]
        col = next(colors)
        ax.plot(func_x, func_y, label="Ilość neuronów {0}".format(neur),
                color="red")
        # plot rbfs
        for neuron_index in range(nn.h_nodes):
            func_y = [nn.query_one([i], int(neuron_index)) for i in func_x]
            ax.plot(func_x, func_y, linestyle="--", color="black")

    plt.grid()
    plt.xlim([-6.5, 6])
    plt.ylim([-7, 4.4])
    plt.title(title)
    # Legend
    plt.legend(loc='upper center',
               ncol=3, fancybox=True, shadow=True)
    plt.savefig("./showRBF/" + title + ".png")


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


number_of_iteration = 150
iter_to_plot = np.linspace(0, number_of_iteration, 6)


input_nodes = 1

output_nodes = 1
learningrate = 0.001
c_range = [min(train_input_list), max(train_input_list)]
sigmas = [[0.1, 0.01], [0.5, 0.5], [8, 10]]

for sig in sigmas:
    # nnworks to plot
    for h_nodes in [21, 41]:
        nnworks = {}
        hidden_nodes = h_nodes
        sig_range = sig
        nn = RBF(input_nodes, hidden_nodes, output_nodes,
                 learningrate, c_range, sig_range)
        nn.set_up_centers_from_vec(train_input_list)

        for i in range(number_of_iteration):
            f = nn.query
            for j in range(len(train_input_list)):
                nn.train_lin(train_input_list[j], train_target_list[j])

        n = copy.deepcopy(nn)
        nnworks[hidden_nodes] = n

        plotChart(train_input_list, train_target_list,
                  test_input_list, test_target_list,
                  nnworks,
                  "Ilosc_neurnow: plik:{1}, Epoki nauki:{0} Neurony: {2} sig_range: {3} "
                  .format(number_of_iteration, str(sys.argv[1]),
                          hidden_nodes, sig_range))
