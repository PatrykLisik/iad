import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import logging
from logging.config import fileConfig
from Perceptron import Perceptron

# logger set up
fileConfig('logging_config.ini')
logger = logging.getLogger()

# points
PointsX = [-0.2, 0.4, 0.6, 1.2, 1.9, 0.5]
PointsY = [0, 0, 1, 1, 1, 0]

# Set up plot
plt.figure(figsize=(10, 10))
plt.xlim([-3, 3])
plt.ylim([-0.5, 1.5])
plt.grid()
plt.title("")
plt.scatter(PointsX, PointsY, color='black', s=10)

# Array with number of iteration when approximation is plotted
iteration_to_plot = [0, 10, 20, 100, 10000, 100000]
# Colors of plots
colors = iter(cm.rainbow(np.linspace(0, 1, len(iteration_to_plot))))

perceptron = Perceptron(1)
x = np.arange(-3, 3, 0.01)

max_iteration_number = 600000
for k in range(max_iteration_number):
    for val, expected_out in zip(PointsX, PointsY):
        perceptron.updateWeigths(val, expected_out)
    if k in iteration_to_plot:
        y1 = perceptron.map(x)
        c = next(colors)
        plt.plot(x, y1, color=c, label="After {0} iterations".format(k))
        logger.info("Iteration {} has been plotted".format(k))

y1 = perceptron.map(x)
plt.plot(x, y1, color='purple', label="final")
legend = plt.legend(loc='best',
                    ncol=3, fancybox=True, shadow=True)
plt.savefig("show_learn.png")
