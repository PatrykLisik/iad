import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

#points
PointsX=[-0.2, 0.4, 0.6, 1.2, 1.9, 0.5]
PointsY=[0, 0, 1, 1, 1, 0]
dimm_number=1 #number of dimmensions
W=np.random.rand(1,dimm_number)
X=np.random.rand(1,dimm_number)
w0=np.random.rand()
f=lambda X:1/(1+np.exp(np.inner(X,W)+w0))
grad=lambda X,y:(f(X)-y)*f(X)*(1-f(X))
print("W:",W,"\nX:",X,"\ngrad",grad(-0.2,0))

x = np.arange(-3, 3, 0.01);
y1=[]
for i in x:
    y1.append(f(i)[0][0])

plt.plot(x,y1,color='red')

itr=2000000
for k in range(itr):
    for i in range(len(PointsX)):
        W+=(grad(PointsX[i],PointsY[i])*PointsX[i])
        w0+=grad(PointsX[i],PointsY[i])
    if (k %(itr/20))==0 or k==[1,10,100]:
        print(W,w0)
        y1=[]
        for j in x:
            y1.append(f(j)[0][0])
        plt.plot(x,y1,color='blue')
print(W,w0)
y1=[]
for j in x:
    y1.append(f(j)[0][0])
plt.plot(x,y1,color='green')
#range
plt.xlim([-3,3])
plt.ylim([-0.5,1.5])
plt.grid()
plt.title("Rozgrzewka 2")
plt.scatter(PointsX,PointsY,color='black', s=10)
plt.savefig("wykres.png")
