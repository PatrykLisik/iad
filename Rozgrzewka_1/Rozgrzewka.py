import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#tex set up
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#range
plt.xlim([-3,3])
plt.ylim([-0.5,1.5])
#points
PointsX=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5]
PointsY=[0, 0, 1, 1, 1, 0]
plt.scatter(PointsX,PointsY,color='black', s=10)
plt.title("Rozgrzewka 1")
x = np.arange(-3, 3, 0.01);
y1 = 1/(1+np.exp(-x))
y2 = 1/(1+np.exp(4*x))
y3 = 1/(1+np.exp(-100*(x-0.55)))
plt.plot(x,y1,color='red', label=r"$f(x)=\frac{1}{e^{-x}}$")
plt.plot(x,y2,color='blue',label=r"$f(x)=\frac{1}{e^{4x}}$")
plt.plot(x,y2,color='green',label=r"$f(x)=\frac{1}{e^{-100*(x-0.55)}}$")

legend = plt.legend(loc='upper center',
          ncol=3, fancybox=True, shadow=True)

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')
plt.savefig("wykres.png")
