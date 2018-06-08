import matplotlib.pyplot as plt
import numpy as np

from functions import Euklides_dist, gaussRad
from RBF import RBF

r = RBF(1, 5, 1)


def foo(x):
    c = 5
    w = 0.5
    sig = 0.2
    # return (gaussRad(Euklides_dist(x, c), sig) * w) + 2
    return r.query(x)


def MSE(f, intput, output):
    ans = 0
    for i in range(len(output)):
        ans += ((f(intput[i])[0] - output[i])**2)
    return np.sum(ans) / len(output)


net = RBF(1, 10, 1)
plt.grid()
x_all = np.linspace(-6, 6, 200)

y_f = [foo(x) for x in x_all]
y_all = [net.query(x) for x in x_all]
y_3 = [net.query(x)[0] for x in x_all]
plt.plot(x_all, y_3, label="rand")

for i in range(500):
    for x, y in zip(x_all, y_f):
        net.train(x, y)
    if i % 50 == 0:
        y_3 = [net.query(x)[0] for x in x_all]
        plt.plot(x_all, y_3, label="train" + str(i))
    if i % 10 == 0:
        y_3 = [net.query(x)[0] for x in x_all]
        print("ERRR: ", MSE(net.query, x_all, y_all))
        c = [x.c for x in net.rbf.rbf]
        sig = [x.sig for x in net.rbf.rbf]
        w = [x.w for x in net.rbf.rbf]
        print("C: ", c)
        print("sig: ", sig)
        print("w: ", w)
        print("Lin", net.out.weigths)
        print("RBF_bias", net.rbf.bias_rbf)


plt.plot(x_all, y_f, label="test", color="black")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.savefig("out2")
plt.show()
