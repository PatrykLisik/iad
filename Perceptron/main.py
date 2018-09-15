#!/usr/bin/env python3
import csv
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from logging.config import fileConfig
from Perceptron import Perceptron


def mse(f, input, output):
    """
    Mean squared error
    """
    ans = 0
    for i in range(len(output)):
        ans += ((f(input[i]) - output[i]) ** 2)
    return ans / len(output)


def get_data(input_file):
    reader = csv.reader(input_file)
    out_x = []
    out_y = []
    for row in reader:
        i = list(map(float, row[0].split(";")))
        out_x.append(i[0:-1])
        out_y.append(i[-1])
    return out_x, out_y


def gen_name(input_file_name):
    no_ext = input_file_name.split(".")[0]
    return "quality" + no_ext[-1] + ".png"


def log_ticks(x, pos):
    if x <= 0:
        return "$0$"
    exponent = int(np.log10(x))
    coefficient = x / 10 ** exponent
    return r"${:2.1f} \times 10^{{ {:2d} }}$".format(coefficient, exponent)


def plot_chart(x, y, out):
    # Set up plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_ticks))
    plt.grid()
    plt.title(gen_name(sys.argv[1]))
    plt.yscale("log")
    plt.xlabel("Number of iterations", fontsize="xx-large")
    plt.ylabel("Mean squared error", fontsize="xx-large")
    ax.plot(x, y, color='purple')
    plt.savefig(out)


if __name__ == '__main__':
    fileConfig('logging_config.ini')
    logger = logging.getLogger()
    input_file = open(sys.argv[1], "r+")
    PointsX, PointsY = get_data(input_file)
    input_file.close()

    perceptron = Perceptron(len(PointsX[0]))

    error = 1
    err_y = []
    err_x = []
    iter_number = 0
    min_error_threshold = 10 ** (-4)
    while error > min_error_threshold:
        error = mse(perceptron, PointsX, PointsY)
        for x, y in zip(PointsX, PointsY):
            perceptron.updateWeigths(x, y)

        if iter_number % 1000 == 0:
            err_y.append(error)
            err_x.append(iter_number)
            logger.info("{} {}".format(iter_number, error))
        iter_number += 1

    print(perceptron.w0[0], file=open(sys.argv[2], "w"))
    for i in perceptron.W:
        # output file name is in second argument
        for j in i:
            print(j, file=open(sys.argv[2], "a"))

    # plot chart
    plot_chart(err_x, err_y, gen_name(sys.argv[1]))
