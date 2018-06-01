from SOM.Neuron_gas import Neuron_gas
from SOM.Kohonen_network import Kohonen_network
from SOM.functions import RNF, GNF, Euklides_dist, quantization_error3
from points_distributions import circumference_dist, triangle_dist, cirlce_dist
from points_distributions import square_dist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from SOM.K_means import K_menas
import copy


def animate(num, data, line, scat, line_sw):
    # line.set_data(np.arange(0,num,0.01),np.sin(np.arange(0,num,0.01)))
    # plt.cla()
    plt.xlabel("Iteracja-{0}".format(num))
    scat.set_offsets(data[num])

    if line_sw:
        # plot lines
        i1 = iter(data[num])
        i2 = iter(data[num])
        next(i2)
        line_x = []
        line_y = []
        for point, next_point in zip(i1, i2):
            x1, y1 = zip(point)
            x2, y2 = zip(next_point)
            line_x.append([x1, x2])
            line_y.append([y1, y2])
        line.set_data(line_x, line_y)

    return (line,)


def plotPointsOfDict(black, redPointsInTime, line_sw, out, title):
    # colors=iter(cm.Set1(np.linspace(0,1,len(data))))
    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    l, = plt.plot([], [], 'r-')
    plt.grid()
    # split black points into x and y
    black_x, black_y = zip(*black)
    ax.scatter(black_x, black_y, color='black', s=10)
    scat = ax.scatter([], [], color='red', s=50)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title(title)
    anim = animation.FuncAnimation(fig, animate, len(redPointsInTime),
                                   fargs=(redPointsInTime, l, scat, line_sw),
                                   interval=1000, blit=True)
    anim.save(out + ".gif", writer='imagemagick')


def plotQError(y, out, title):
    # assume that one y in one iteration
    x = range(len(y))
    # Set up plot
    plt.figure(figsize=(10, 10))
    plt.xlabel("Numer itercji")
    plt.ylabel("Błąd kwantyzacji")
    plt.yscale("log")
    plt.grid()
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(out + ".png")


def point_set_up_1():
    aprox_set = cirlce_dist([-5, 5], 2, 200)
    aprox_set.extend(triangle_dist([8, 8], [8, 5], [6, 6], 200))
    aprox_set.extend(square_dist([- 5, -5], 3, 300))
    aprox_set.extend(circumference_dist([5, -5], 4, 100))
    aprox_set.extend(circumference_dist([0, 1], 2, 100))
    # aprox_set.extend(triangle_dist([2, -2], [5, -9], [8, -3], 60))
    return aprox_set


def point_set_up_2():
    aprox_set = []
    aprox_set.extend(circumference_dist([0, 0], 8, 300))
    aprox_set.extend(circumference_dist([0, 0], 4, 150))
    # aprox_set.extend(circumference_dist([0, 0], 4.5, 150))
    # aprox_set.extend(circumference_dist([0, 0], 5, 150))
    # aprox_set.extend(circumference_dist([0, 0], 5.5, 150))
    return aprox_set


def point_set_up_3():
    aprox_set = cirlce_dist([-7, 7], 2, 200)
    aprox_set.extend(triangle_dist([8, 8], [8, 5], [6, 6], 50))
    aprox_set.extend(circumference_dist([- 5, -5], 3, 40))
    aprox_set.extend(square_dist([5, -5], 2, 300))
    aprox_set.extend(circumference_dist([0, 1], 2, 50))
    return aprox_set


def deadNumber(data):
    data_trans = np.swapaxes(data, 1, 0)
    ans = 0
    # print("data", data)
    for neuron_history in data_trans:
        # test if all are the same
        ans += int((neuron_history[1:] == neuron_history[:-1]).all())
    return ans


# aprox_set = [point_set_up_1(), point_set_up_2(), point_set_up_3()]
# aprox_set = [point_set_up_3()]
aprox_set = [circumference_dist([0, 0], 7, 600)]
R = 3
for set_nr, set in enumerate(aprox_set):
    for points_number in [12, 18, 20]:
        kn = Kohonen_network(points_number, set)
        ng = Neuron_gas(points_number, set)
        km = K_menas(points_number, set)
        neuronPosList_koh = []
        neuronPosList_gas = []
        neuronPosList_km = []
        Qerr_koh = []
        Qerr_gas = []
        Qerr_Km = []
        print("Compute")
        for i in range(7):
            print("IERT: ", i)
            # kohonen_network
            values_kn = list(kn.neurons.values())
            neuronPosList_koh.append(values_kn)
            Qerr_koh.append(quantization_error3(values_kn, set, Euklides_dist))
            kn.iter_once()
            # neuron gas
            values_ng = list(ng.neurons.values())
            neuronPosList_gas.append(values_ng)
            Qerr_gas.append(quantization_error3(values_ng, set, Euklides_dist))
            ng.iter_once()
            # print(values)

            values_km = copy.deepcopy(list(km.getNeurons()))
            neuronPosList_km.append(values_km)
            Qerr_Km.append(quantization_error3(values_km, set, Euklides_dist))
            km.iter_once()
        print("{} neuron".format(points_number))
        print("DEAD KN", deadNumber(neuronPosList_koh))
        print("DEAD NG", deadNumber(neuronPosList_gas))
        print("DEAD KM", km.dead_neurons)
        print(neuronPosList_km, file=open(
            "Kmeans{}.txt".format(points_number), "w"))
        plotPointsOfDict(set, neuronPosList_koh, True,
                         "koh-GNF-Points={0}:set={1}".
                         format(points_number, set_nr),
                         "Sieć Kohonena - {} punktów".format(points_number))
        plotPointsOfDict(set, neuronPosList_gas, False,
                         "gas-GNF-Points={0}:set={1}:R={2}".
                         format(points_number, set_nr, R),
                         "Gas neuronowy - {} punktów".format(points_number))
        plotPointsOfDict(set, neuronPosList_km, False,
                         "k_means-GNF-Points={0}:set={1}:R={2}".
                         format(points_number, set_nr, R),
                         "K-Srednie - {} punktów".format(points_number))
        plotQError(Qerr_gas, "Qerr2-gas-GNF-Points={0}:set={1}:R={2}".
                   format(points_number, set_nr, R),
                   "Gas neuronowy - bład - {} punktów".format(points_number))
        plotQError(Qerr_koh, "Qerr2-koh-GNF-Points={0}:set={1}:R={2}".
                   format(points_number, set_nr, R),
                   "Sieć Kohonena - bład - {} punktów".format(points_number))
        plotQError(Qerr_Km, "Qerr2-kmeans-Points={0}:set={1}:R={2}".
                   format(points_number, set_nr, R),
                   "K-Srednie - bład - {} punktów".format(points_number))
