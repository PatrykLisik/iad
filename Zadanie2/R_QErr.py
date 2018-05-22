import matplotlib.pyplot as plt
from SOM.Kohonen_network import Kohonen_network as KN
from SOM.Neuron_gas import Neuron_gas as NG
from points_distributions import cirlce_dist, circumference_dist, square_dist
from SOM.functions import quantization_error2 as QErr
from SOM.functions import Euklides_dist as E_dist
import numpy as np
from matplotlib.pyplot import cm


def plotQError(y_tab, out, title):
    colors = iter(cm.rainbow(np.linspace(0, 1, len(y_tab))))
    # Set up plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plt.xlabel("Numer itercji")
    plt.ylabel("Błąd kwantyzacji")
    plt.yscale("log")
    plt.grid()
    neuron_number = 1
    for y in y_tab:
        x = range(len(y))
        plt.plot(x, y, color=next(colors),
                 label="R {}".format(neuron_number))
        neuron_number += 1
    plt.title(title)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=5)
    plt.savefig(out + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')


# set = circumference_dist([-4, 4], 3, 100)
obj = [circumference_dist([0, 0], 7, 400), cirlce_dist(
    [0, 0], 8, 500), square_dist([0, 0], 5, 600)]
for set, desc in zip(obj, ["circumference", "cirlce", "square"]):
    for som, name in zip([NG], ["Gas neuronowy"]):
        data = []
        neuron_number = 20
        for R in range(1, 8, 1):
            q_errs = []
            net = som(neuron_number, set, neighborhood_radius=R)
            print("R: ", R)
            for _ in range(15):
                err = QErr(net.neurons.values(), set, E_dist)
                q_errs.append(err)
                net.iter_once()
            data.append(q_errs)
        plotQError(data, "R" + desc + " " + name, name)
