import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def driver():
    n = 2 ##Find P_n(x) roots
    an, bn, cn = generate_tchebycoeffs(n+1)
    print(an, bn, cn)
    roots = rootfind_alg2(an, bn, cn)
    print(roots)

    xeval = np.linspace(-2, 2, 100)  # vector of x
    P = cheb(n+1, xeval)
    plt.figure()
    plt.plot(xeval,P[n+1],label = f'P_{n}')
    plt.title(f'P_{n}')
    for root in roots:
        plt.scatter(root, y= 0,  color = 'black', s = 50, marker='x', label = 'x = ' + str(root))
    plt.legend()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()


def rootfind_alg2(an, bn, cn):
    n = len(an)
    alpha = np.array([-bn[i] / an[i] for i in range(n)])
    beta = np.array([np.sqrt(cn[i+1] / (an[i] * an[i+1])) for i in range(n - 1)])
    J = np.zeros((n, n))
    np.fill_diagonal(J, alpha)
    for i in range(n - 1):
        J[i, i + 1] = beta[i]
        J[i + 1, i] = beta[i]
    print(J)
    eigvals = np.linalg.eigvalsh(J)
    return np.sort(eigvals)


def generate_tchebycoeffs(n):
    # For Chebyshev polynomials of the first kind
    an = np.ones(n)*2
    an[0] = 1   
    bn = np.zeros(n)         
    cn = np.ones(n)           
    return an, bn, cn


def QR(J):
    max_iter = 1000
    tol = 1e-12
    for _ in range(max_iter):
        Q, R = np.linalg.qr(J)
        J_new = R @ Q
        if np.allclose(J, J_new, atol=tol):
            break
        J = J_new
    return J


def cheb(n, xeval):
    T = np.zeros((n+1, len(xeval)))  # 2d array holds n+1 arrays of xevals
    T[0, :] = 1  # T0(x) = 1
    if n > 0:
        T[1, :] = xeval  # T1(x) = x
    
    for i in range(2, n+1): # applying recurrence relation
        T[i, :] = 2 * xeval * T[i-1, :] - T[i-2, :]
    return T

driver()
