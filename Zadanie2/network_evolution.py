import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from scipy.spatial import Voronoi

from points_distributions import points as data_set
from SOM.functions import voronoi_finite_polygons_2d
from SOM.K_means import K_means as KM
from SOM.Kohonen_network import Kohonen_network as KN
from SOM.Neuron_gas import Neuron_gas as NG


def plot(black, redPointsInTime, out, title):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(redPointsInTime))))
    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    l, = plt.plot([], [], 'r-')
    plt.grid()
    # split black points into x and y
    black_x, black_y = zip(*black)
    ax.scatter(black_x, black_y, color='black', s=7)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title(title)

    i1 = iter(redPointsInTime)
    i2 = iter(redPointsInTime)
    next(i2)
    # print("redPointsInTime", redPointsInTime)
    for nr, (xy_cur, xy_next) in enumerate(zip(i1, i2)):
        xs_cur, ys_cur = zip(*xy_cur)
        xs_next, ys_next = zip(*xy_next)
        color = next(colors)
        plt.plot([], [], color=color, label="iteracja {}-{}".format(nr, nr + 1))
        plt.scatter(xs_cur, ys_cur, color="red", s=20)
        plt.scatter(xs_next, ys_next, color="red", s=20)
        for x_cur, y_cur, x_next, y_next in zip(xs_cur, ys_cur, xs_next, ys_next):
            dx = x_next - x_cur
            dy = y_next - y_cur
            if(dx != 0 or dy != 0):
                plt.arrow(x_cur, y_cur, dx, dy,
                          color=color, head_width=0.15, head_length=0.25,
                          length_includes_head=True)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=5)
    plt.savefig(out + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')


def plotVoronoi(black, redPointsInTime, out, title, koh_line_swt):

    red_at_last = list(redPointsInTime[-1])
    vor = Voronoi(red_at_last)
    # plot
    fig = plt.figure(figsize=(7, 7))
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)
    # plot lines beteeen kononen network
    if koh_line_swt:
        print("koh plt")
        red_iter = iter(red_at_last)
        red_iter_next = iter(red_at_last)
        next(red_iter_next)
        for point, point_next in zip(red_iter, red_iter_next):
            plt.plot([point[0], point_next[0]], [
                     point[1], point_next[1]], 'r--')

    for p in red_at_last:
        plt.scatter(p[0], p[1], color='red', s=10)

    plt.grid()
    # split black points into x and y
    black_x, black_y = zip(*black)
    plt.title(title)
    plt.scatter(black_x, black_y, color='black', s=1)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.savefig(out + ".png")


Neuron_number = 10
som = {"Gas_Neuronowy": NG,
       "Siec_kohonena": KN,
       "k-Srednie": KM}
iter_number = 10
for som_name, SOM_type in som.items():
    for points_name, points in data_set.items():
        map = SOM_type(Neuron_number, points)
        redInTime = []
        for _ in range(iter_number):
            neuron_pos = list(map.getNeurons())
            redInTime.append(neuron_pos)
            map.iter_once()
        koh = False
        if som_name == "Siec_kohonena":
            koh = True
            print("kon " + som_name)
        plotVoronoi(points, redInTime, "./network_evo/" +
                    som_name + "_" + points_name + "_vor", som_name, koh)
        plot(points, redInTime, "./network_evo/" +
             som_name + "_" + points_name + "_in_time", som_name)
