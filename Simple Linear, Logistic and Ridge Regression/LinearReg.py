# -*- coding: utf-8 -*-
"""
@author: Achyutha Sreenivasa Bharadwaj
@asuid: 1213094192
Python 3.6.3
Created on Sun Jan 28 18:48:33 2018
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('linRegData.npy')

X = data[:,0]
Y = data[:,1]

_ones = np.ones(np.size(X,0))

X = np.column_stack((_ones,X))

#theta = ((X'X)^-1).X'Y
theta = np.dot(np.linalg.inv(np.dot(X.T, X)),np.dot(X.T,Y))

#y_cap = X.theta
Y_cap = np.dot(X,theta)

#means square error = mean((Y - Y_cap)^2) 
mean_square_error = ((Y - Y_cap) ** 2).mean()

plt.plot(X[:,1], Y, 'ro')
plt.plot(X[:,1], Y_cap)
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Linear Regression')
plt.savefig('Linear_Reg.png')

print('MSE: {}'.format(mean_square_error))