from SOM.Neuron_gas import Neuron_gas as NG
from SOM.Kohonen_network import Kohonen_network as KN
from points_distributions import circumference_dist, triangle_dist, cirlce_dist
from points_distributions import square_dist
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


def plot(black, redPointsInTime, out, title):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(redPointsInTime))))
    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    l, = plt.plot([], [], 'r-')
    plt.grid()
    # split black points into x and y
    black_x, black_y = zip(*black)
    ax.scatter(black_x, black_y, color='black', s=10)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title(title)

    i1 = iter(redPointsInTime)
    i2 = iter(redPointsInTime)
    next(i2)
    #print("redPointsInTime", redPointsInTime)
    for nr, (xy_cur, xy_next) in enumerate(zip(i1, i2)):
        label = "Iteracja nr {}".format(nr)
        xs_cur, ys_cur = zip(*xy_cur)
        xs_next, ys_next = zip(*xy_next)
        color = next(colors)
        plt.plot([], [], color=color, label="iteracja {}".format(nr))
        plt.scatter(xs_cur, ys_cur, color="red", s=20)
        plt.scatter(xs_next, ys_next, color="red", s=20)
        for x_cur, y_cur, x_next, y_next in zip(xs_cur, ys_cur, xs_next, ys_next):
            plt.plot([x_cur, x_next], [y_cur, y_next],
                     color=color)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=5)
    plt.savefig(out + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')


points = circumference_dist([0, 0], 7, 400)

Neuron_number = 20
som = {"Gas_Neuronowy": NG,
       "Siec_kohonena": KN}
iter_number = 6
for decription, type in som.items():
    map = type(Neuron_number, points)
    redInTime = []
    for _ in range(iter_number):
        neuron_pos = list(map.neurons.values())
        redInTime.append(neuron_pos)
        map.iter_once()
    plot(points, redInTime, decription + "_in_time", decription)
