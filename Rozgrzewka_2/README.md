# Rozgrzewka 2

## Abstract
Approximate n-dimensional sigmoid function to given points with gradient descent

![equation](http://latex.codecogs.com/gif.latex?f(X)=\frac{1}{1&plus;exp(w_nx_n&plus;w_{n-1}x_{n-1}&plus;...&plus;w_1x_1&plus;w_0)}=\frac{1}{1&plus;exp(W\cdot&space;X&plus;w_0)})


where

 ![equation](http://latex.codecogs.com/gif.latex?$$X=\begin{bmatrix}&space;x_n&x_{n-1}&space;&...&space;&&space;x_1&space;&&space;\end{bmatrix}$$) is a point i n-dimensional space


![equation](http://latex.codecogs.com/gif.latex?$$W=\begin{bmatrix}&space;w_n&w_{n-1}&space;&...&space;&&space;w_1&space;&&space;\end{bmatrix}$$) is vector of weights

![equation](http://latex.codecogs.com/gif.latex?$$w_0$$) can be considered as bias

## Solution
 Let denote error function

  ![equation](http://latex.codecogs.com/gif.latex?h(X)=\frac{1}{2n}\sum_{n}^{i=1}(f(X^j)-y^j)^2)

 where

 ![equation](http://latex.codecogs.com/gif.latex?(X^j,y^j)) is given point

To find best weights, it is necessary to find global minimum, which is not a feasible thing to do. Decent result can be achieved by finding a local minimum.

![equation](http://latex.codecogs.com/gif.latex?\large&space;\nabla&space;h(X)=\begin{bmatrix}&space;\frac{\partial&space;h(X)}{\partial&space;x_{n-1}}&space;\\&space;\frac{\partial&space;h(X)}{\partial&space;x_{n-2}}\\&space;\vdots\\&space;\frac{\partial&space;h(X)}{\partial&space;x_{1}}\\&space;\end{bmatrix})

Let $$\text{S}_1(N) = \sum_{p=1}^N \text{E}(p)$$

![equation](http://bit.ly/2Hp8sCT) 

 ![equation](http://latex.codecogs.com/gif.latex?\dpi{120}&space;\large&space;\nabla&space;h(X)=\begin{bmatrix}&space;\frac{1}{n}\sum_{n}^{i=1}(f(X^j)-y^j)f(X^j)(1-f(X^j))x_{n}&space;\\&space;\frac{1}{n}\sum_{n}^{i=1}(f(X^j)-y^j)f(X^j)(1-f(X^j))x_{n-1}\\&space;\vdots\\&space;\frac{1}{n}\sum_{n}^{i=1}(f(X^j)-y^j)f(X^j)(1-f(X^j))x_1\\&space;\end{bmatrix})
