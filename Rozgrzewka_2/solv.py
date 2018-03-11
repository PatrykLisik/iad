import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

class Aprox(object):
    def __init__(self, dimm_number):
        self.W=np.random.rand(1,dimm_number)
        self.w0=np.random.rand()
        #step aka learnig rate
        self.step=0.1
    def sigmoid(self,X):
        #Muliti-dimensional sigmoid
        #output is matrix is 1x1,
        return 1/(1+np.exp(np.inner(X,self.W)+self.w0))
    def grad(self,X,y):
        #Notice that it is not multiplay by x_i
        f=self.sigmoid
        return (f(X)-y)*f(X)*(1-f(X))
    def __call__(self,X):
        #output of sigmoid is matrix is 1x1, so one value is returned
        return self.sigmoid(X)[0][0]
    def updateWeigths(self,X,y):
        #basicly learnig
        g=self.grad(X,y)*self.step
        self.W+=g*X;
        self.w0+=g
    def map(self,x):
        ans=[]
        for i in x:
            ans.append(self.__call__(i))
        return ans



#points
PointsX=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5]
PointsY=[0, 0, 1, 1, 1, 0]

plt.figure(figsize=(10,10))
#range
plt.xlim([-3,3])
plt.ylim([-0.5,1.5])
plt.grid()
plt.title("Rozgrzewka 2")
plt.scatter(PointsX,PointsY,color='black', s=10)

#Arry with number of iteration when aproximation is plotted
iterToPlot=[0,10,20,100,10000,100000]
#Colors of plots
colors=iter(cm.rainbow(np.linspace(0,1,len(iterToPlot))))

aprox=Aprox(1)
x = np.arange(-3, 3, 0.01);

itr=500000
for k in range(itr):
    for i in range(len(PointsX)):
        aprox.updateWeigths(PointsX[i],PointsY[i])
    if k in iterToPlot:
        y1=aprox.map(x)
        c=next(colors)
        plt.plot(x,y1,color=c, label="After {0} iterations".format(k))

y1=aprox.map(x)
plt.plot(x,y1,color='pink',label="final")
legend = plt.legend(loc='best',
          ncol=3, fancybox=True, shadow=True)
plt.savefig("wykres.png")
