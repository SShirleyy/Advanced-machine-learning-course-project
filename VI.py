# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:16:39 2018

@author: SHE
"""

from __future__ import division
import numpy as np
from scipy.stats import norm, gamma 
import matplotlib.pyplot as plt

#parameter set
mu_0 = 0
lambda_0 = 0
a_0 = 10
b_0 = 100
N = 1000
#generate data
X = np.random.normal(0,1,N)
mean = np.mean(X)

#compute parameter
mu_N = (lambda_0 * mu_0 + N * mean)/(lambda_0 +N)
a_N = a_0 + N/2
#initialize parameter
lambda_N = 0.2
#updata parameter
sum_square = np.sum(X ** 2)
X_sum = sum(X)

b_N = b_N_old = 0

epsilon = 0.000001
maxIter = 100
for i in range(maxIter):
    b_N_old = b_N
    b_N = b_0 + 0.5*((lambda_0 + N)*(1/lambda_N + mu_N **2)-2*(lambda_0 * mu_0 + X_sum) *mu_N +sum_square + lambda_0 * mu_0**2) 
    lambda_N = (lambda_0 + N) * (a_N / b_N)
    # print(lambda_N, b_N)
    if abs(b_N - b_N_old) < epsilon:
        break
    
# exact posterior
mu_e = (X_sum + lambda_0 * mu_0)/(lambda_0 + N)
lambda_e = N + lambda_0 
a_e = (N+1)/2. +a_0
X_sub = X - mean
X_squre = X_sub ** 2
b_e = b_0 + 0.5*(np.sum(X_squre) + (N*lambda_0*((mean - mu_0) ** 2))/(N + lambda_0))

print('e',  mu_e, lambda_e, a_e, b_e)
print('a', mu_N, lambda_N, a_N, b_N)

x_e = np.linspace(0.8, 1.2, 1000)
Norm_e = norm.pdf(x_e, mu_e, 1/lambda_e)
y_e = np.linspace(0.01, 0.02, 1000)
Gamma_e = gamma.pdf(y_e, a_e, scale = 1/b_e)

x = np.linspace(0.8, 1.2, 1000)
Norm = norm.pdf(x, mu_N, 1/lambda_N)
y = np.linspace(0.01, 0.02, 1000)
Gamma = gamma.pdf(y, a_N, scale = 1/b_N)

Z = np.outer(Gamma, Norm)
Z_norm = Z / np.sum(Z)

Z_e= np.outer(Gamma_e, Norm_e)
Z_norme = Z_e / np.sum(Z_e) 
plt.contourf(x, y, Z_norm, 20, alpha = 0.75, cmap = plt.cm.hot)
plt.xlabel( 'mu ' , fontsize = 15)
plt.ylabel( 'tau' , fontsize = 15)
plt.show()

plt.contourf(x_e, y_e, Z_norme, 20, alpha = 0.75, cmap = plt.cm.hot)
plt.xlabel( 'mu ' , fontsize = 15)
plt.ylabel( 'tau' , fontsize = 15)
plt.show()
