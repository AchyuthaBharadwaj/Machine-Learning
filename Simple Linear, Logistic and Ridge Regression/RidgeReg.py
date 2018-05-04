92# -*- coding: utf-8 -*-
"""
@author: Achyutha Sreenivasa Bharadwaj
@asuid: 1213094192
Python 3.6.3
Created on Sun Jan 28 18:48:33 2018
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('linRegData.npy')
np.random.shuffle(data)

X = data[:,0]
Y = data[:,1]

phi_X = np.array([np.power(x, np.arange(16)) for x in X])

lambdas = np.array([0.01,0.05,0.1,0.5,1.0,5,10])
avg_cv_errors = []
avg_train_errors = []

cross_folds = 10
test_size = int(np.size(X,0)/cross_folds)

for l in lambdas:
    cv_error = 0
    train_error = 0
    for i in range(0,cross_folds):
        x_test = phi_X[i*test_size:i*test_size+test_size,:]
        x_train = np.concatenate((phi_X[:i*test_size,:], phi_X[i*test_size+test_size:,:]))
        y_test = Y[i*test_size:i*test_size+test_size]
        y_train = np.concatenate((Y[:i*test_size], Y[i*test_size+test_size:]))
        theta = np.dot(np.linalg.inv(np.dot(x_train.T, x_train) + np.identity(np.size(x_train,1))*l*l),
                       np.dot(x_train.T,y_train))
        y_test_cap = np.dot(x_test,theta)
        y_train_cap = np.dot(x_train,theta)
        cv_error += ((y_test - y_test_cap) ** 2).mean()
        train_error += ((y_train - y_train_cap) ** 2).mean()
    avg_cv_errors.append(cv_error/cross_folds)
    avg_train_errors.append(train_error/cross_folds)
    
plt.plot(lambdas, avg_cv_errors)
plt.title('Lambda vs Validation Error')
plt.ylabel('Validation Error')
plt.xlabel('Lambda')
plt.savefig('Ridge_Regression_LambdaVsValidationErrors.png')
plt.show()
plt.plot(lambdas, avg_train_errors)
plt.ylabel('Train Error')
plt.xlabel('Lambda')
plt.title('Lambda vs Train Error')
plt.savefig('Ridge_Regression_LambdaVsTrainErrors.png')
plt.show()

min_l_pos = np.where(avg_cv_errors == np.min(avg_cv_errors))[0][0]
opt_lambda = lambdas[min_l_pos]

data = np.load('linRegData.npy')
X = data[:,0]
Y = data[:,1]

phi_X = np.array([np.power(x, np.arange(16)) for x in X])

theta = np.dot(np.linalg.inv(np.dot(phi_X.T, phi_X) + np.identity(np.size(phi_X,1))*opt_lambda*opt_lambda),
               np.dot(phi_X.T,Y))
Y_cap = np.dot(phi_X,theta)

plt.plot(X, Y, 'ro')
plt.plot(X, Y_cap)
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Ridge Regression')
plt.savefig('Ridge_Regression.png')