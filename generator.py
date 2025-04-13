import numpy as np
import matplotlib.pyplot as plt

def driver():



    n = 5 # number of polynomials to generate
    np1=3  # the n plus 1 polynomial you want graphed, it will graph np1 and np1 -1 polynomials

    xeval = np.linspace(-2, 2, 100)  # vector of x
    plotter(cheb(n,xeval),np1,xeval) # plots the chebyshev 
    plotter(hermite(n,xeval),np1,xeval) # plots the hermite 

def plotter(P,np1,xeval):
    plt.figure()
    plt.plot(xeval,P[np1],label = f'P_{np1}')
    plt.plot(xeval,P[np1-1], label = f'P_{np1-1}')
    plt.legend()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()



def cheb(n, xeval):
    T = np.zeros((n+1, len(xeval)))  # 2d array holds n+1 arrays of xevals
    T[0, :] = 1  # T0(x) = 1
    if n > 0:
        T[1, :] = xeval  # T1(x) = x
    
    for i in range(2, n+1): # applying recurrence relation
        T[i, :] = 2 * xeval * T[i-1, :] - T[i-2, :]
    return T

def hermite(n, xeval):
    H = np.zeros((n+1, len(xeval)))
    H[0, :] = 1  # H0(x) = 1
    if n > 0:
        H[1, :] = xeval  # H1(x) = x
    
    for i in range(2, n+1):
        H[i, :] = 2 * xeval * H[i-1, :] - 2 * (i-1) * H[i-2, :]
    return H

driver()