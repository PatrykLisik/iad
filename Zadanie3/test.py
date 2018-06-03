import matplotlib.pyplot as plt
import numpy as np
from RBF import RBF

net = RBF(1, 4, 1)
plt.grid()
x_all = np.linspace(0, 10, 1000)
y_all = [net.query(x)[0] for x in x_all]
plt.plot(x_all, y_all, color="black")

plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("out1")
plt.show()
