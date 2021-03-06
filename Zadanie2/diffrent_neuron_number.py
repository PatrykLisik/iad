import matplotlib.pyplot as plt
from SOM.Kohonen_network import Kohonen_network as KN
from SOM.Neuron_gas import Neuron_gas as NG
from SOM.K_means import K_means as KM
from points_distributions import points
from SOM.functions import quantization_error3 as QErr
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
    neuron_number = 2
    for y in y_tab:
        x = range(len(y))
        plt.plot(x, y, color=next(colors),
                 label="Ilość neuronów {}".format(neuron_number))
        neuron_number += 2
    plt.title(title)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=5)
    plt.savefig(out + ".png", bbox_extra_artists=(lgd,), bbox_inches='tight')


# set = circumference_dist([-4, 4], 3, 100)
obj = points

networks = {KN: "Siec_Kohonena",
            NG: "Gas_neuronowy",
            KM: "k-Srednie"}


for desc, set in obj.items():
    for som, name in networks.items():
        data = []
        for neuron_number in range(2, 21, 2):
            q_errs = []
            net = som(neuron_number, set)
            print("neuron_number: ", neuron_number)
            for _ in range(25):
                err = QErr(net.getNeurons(), set, E_dist)
                q_errs.append(err)
                net.iter_once()
            data.append(q_errs)
        plotQError(data, "./diffrent_neuron_number/" + desc + "_" + name, name)
