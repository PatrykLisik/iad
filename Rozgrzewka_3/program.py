#!/usr/bin/env python3
import numpy as np
from NeutralNetwork import NeutralNetwork
from matplotlib import ticker
import matplotlib.pyplot as plt
import csv
import sys


def mse(f, input, output):
    ans = 0
    for i in range(len(output)):
        ans += ((f(input[i]).T - output[i]) ** 2)
    return np.sum(ans) / len(output)


def get_data(input_file):
    reader = csv.reader(input_file)
    out_x = []
    out_y = []
    for row in reader:
        row_list = list(map(float, row[0].split(";")))
        out_x.append(row_list[0:-1])
        out_y.append(row_list[-1])
    return out_x, out_y


def gen_name(input_file_name):
    name_array = input_file_name.split(".")[0]
    return "quality" + name_array[-1] + ".png"


def my_ticks(x, pos):
    if x <= 0:
        return "$0$"
    exponent = int(np.log10(x))
    coeff = x / 10 ** exponent
    return r"${:2.1f} \times 10^{{ {:2d} }}$".format(coeff, exponent)


def plot_chart(x, y, out):
    # Set up plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(my_ticks))
    plt.grid()
    plt.title(gen_name(sys.argv[1]))
    plt.yscale("log")
    plt.xlabel("Number of iterations", fontsize="xx-large")
    plt.ylabel("Mean squared error", fontsize="xx-large")
    ax.plot(x, y, color='purple')
    plt.savefig(out)


if __name__ == '__main__':
    input_file = open(sys.argv[1], "r+")
    input_list, target_list = get_data(input_file)
    input_file.close()

    input_nodes = len(input_list[0])
    hidden_nodes = 4
    output_nodes = 1
    learning_rate = 0.1
    nn = NeutralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    error_test = 10
    error_x = []
    error_y = []
    for i in range(6 * 10 ** 5):
        for j in range(len(input_list)):
            nn.train(input_list[j], target_list[j])
        f = nn.query
        error_test = mse(f, input_list, target_list)
        error_x.append(i)
        error_y.append(error_test)
        print(error_test)
        i += 1
    plot_chart(error_x, error_y, gen_name(sys.argv[1]))
