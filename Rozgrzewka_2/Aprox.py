import numpy as np

class Aprox(object):
    def __init__(self, dimm_number):
        self.W=np.random.rand(1,dimm_number)
        self.w0=np.random.rand()
        #step aka learnig rate
        self.step=0.1
    def sigmoid(self,X):
        #Muliti-dimensional sigmoid
        #output is matrix is 1x1,
        return 1/(1+np.exp(-(np.inner(X,self.W)+self.w0)))
    def grad(self,X,y):
        #Notice that it is not multiplay by x_i
        f=self.sigmoid
        return (f(X)-y)*f(X)*(1-f(X))
    def __call__(self,X):
        #output of sigmoid is matrix is 1x1, so one value is returned
        return self.sigmoid(X)[0]
    def updateWeigths(self,X,y):
        #basicly learnig
        g=self.grad(X,y)*self.step
        self.W-=g*X;
        self.w0-=g
    def map(self,x):
        ans=[]
        for i in x:
            ans.append(self.__call__(i))
        return ans
