from SOM.Neuron_gas import Neuron_gas as NG
from SOM.Kohonen_network import Kohonen_network as KN
from SOM.K_means import K_menas as KM
from points_distributions import circumference_dist
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from scipy.spatial import Voronoi
from SOM.functions import voronoi_finite_polygons_2d


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


def plotVoronoi(black, redPointsInTime, out, title):

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

    for p in red_at_last:
        plt.scatter(p[0], p[1], color='red', s=5)

    plt.grid()
    # split black points into x and y
    black_x, black_y = zip(*black)
    plt.title(title)
    plt.scatter(black_x, black_y, color='black', s=1)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.savefig(out + ".png")


points = circumference_dist([0, 0], 8, 601)
# points += square_dist([-5, -5], 3, 200)

Neuron_number = 10
som = {"Gas_Neuronowy": NG,
       "Siec_kohonena": KN,
       "k-Srednie": KM}
iter_number = 5
for decription, SOM_type in som.items():
    map = SOM_type(Neuron_number, points)
    redInTime = []
    for _ in range(iter_number):
        neuron_pos = list(map.getNeurons())
        redInTime.append(neuron_pos)
        map.iter_once()
    plotVoronoi(points, redInTime, decription + "_vor", decription)
    plot(points, redInTime, decription + "_in_time", decription)
