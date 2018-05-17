# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:19:56 2017

@author: SHE
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

####
# Model definition
####

# Generates the probability of each outcome
# for each dice for all tables uniformly using
# the flat Dirichlet distribution [K*2*6]
def generateB(K):
    return np.random.dirichlet(np.ones(6), (K, 2))
#    uniform = np.ones((K, 2, 6)) / 6.
#    return uniform
#    uniform[0] = np.random.dirichlet(np.ones(6), 2)
#    return uniform

# Generates the sequence for how a player
# trasitions to the next table [K]
def generateTableSeq(K):
    tables = [np.random.choice([0, 1])]
    for i in range(K-1):
        prob = 0.25
        if (tables[-1]):
            prob = 0.75
        tables.append(np.random.choice([0, 1], p=[prob, 1-prob]))

    return tables

# One player plays the game with an idividual
# trasition sequence ([K],1)]
Outcome = namedtuple('Outcome', ['X', 'S'])
def playOnce(B, K):
    tableSeq = generateTableSeq(K)
    x = []
    s = 0
    for k in range(K):
        obs = np.random.choice(int(6), p=B[k][tableSeq[k]]) + 1 
        x.append(obs)

    return Outcome(x, s)

# Some observations in O.X are dropped '
# with probability p ([K],1)]
def hideObservations(O, p):
    return Outcome([None if np.random.uniform(0, 1) > p else x for x in O.X], O.S)

# Let (N) players play the game [N*(K,1)]
def play(N, B, K, p):
    return [hideObservations(playOnce(B, K), p) for n in range(N)]

####
# Model Test (Question 4 and Question 5)
####
K = 10   # Num tables
N = 20   # Num players
p = 1.0  # 1-p is dropout probability
iterations = 8000

B = generateB(K)

# Experimental probability
counts = np.zeros((K, 6))
for i in range(iterations):
    obs = play(N, B, K, p)
    for n in range(N):
        for k in range(K):
            if obs[n].X[k]:
                counts[k][obs[n].X[k]-1] += 1

exp_prob = counts / counts.sum(axis=1, keepdims=True)

# Theoretical probabiliy
theo_prob = np.array([(B[k][0]+B[k][1]) / 2. for k in range(K)])

# Plot
plt.hold(True)
plt.plot(exp_prob.flatten(),'ro-')
plt.plot(theo_prob.flatten(), 'bo-')
#plt.axis([0, 6, 0, 1])
plt.ylabel('Probability')
plt.show()

####
# DP implementation (Question 7)
####

# Transition matrix
def generateA():
    return [[1/4., 3/4.], [3/4., 1/4.]]

# Forward algoritm
def forward(K, A, B, X, S):
    alpha = np.zeros(K, 2, 6*K)
    
    # Alpha base case
    if X[0]:
        alpha[0][:,X[0]-1] = 1/2. * B[0][:,X[0]-1]
    else:
        alpha[0][:,:6] = 1/2. * B[0][:,:6] 
    
    # Alpha iteration
    for k in range(1, K):
        if X[0]:
            for s in range(0, 6*K):
                if s > X[k]:  
                    for l in range(2):
                        for i in range(2): 
                            alpha[k][i][s] = alpha[k-1][i][s-X[K]-1] * A[l][i] * B[k][i][X[k]-1]
        else:
            for l in range(2):
                for i in range(2): 
                    for s in range(0, 6*K):
                        alpha[k][i][s] = alpha[k-1][i][s-] * A[l][i] * B[k][i][X[k]-1]
            
def backward():
    return 0

def forwardBackward():
    return 0    

