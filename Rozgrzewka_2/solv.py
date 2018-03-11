import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from Aprox import *


#points
PointsX=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5]
PointsY=[0, 0, 1, 1, 1, 0]

#Set up plot
plt.figure(figsize=(10,10))
plt.xlim([-3,3])
plt.ylim([-0.5,1.5])
plt.grid()
plt.title("")
plt.scatter(PointsX,PointsY,color='black', s=10)

#Arry with number of iteration when aproximation is plotted
iterToPlot=[0,10,20,100,10000,100000]
#Colors of plots
colors=iter(cm.rainbow(np.linspace(0,1,len(iterToPlot))))

aprox=Aprox(1)
x = np.arange(-3, 3, 0.01);

itr=600000
for k in range(itr):
    for i in range(len(PointsX)):
        aprox.updateWeigths(PointsX[i],PointsY[i])
    if k in iterToPlot:
        y1=aprox.map(x)
        c=next(colors)
        plt.plot(x,y1,color=c, label="After {0} iterations".format(k))

y1=aprox.map(x)
plt.plot(x,y1,color='purple',label="final")
legend = plt.legend(loc='best',
          ncol=3, fancybox=True, shadow=True)
plt.savefig("chart1.png")
