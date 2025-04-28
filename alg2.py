import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def driver():
    n = 2 ##Find P_n(x) roots
    an, bn, cn = generate_tchebycoeffs(n+1)
    print(an, bn, cn)
    roots = rootfind_alg2(an, bn, cn)
    print(roots)

def rootfind_alg2(an, bn, cn):
    ##Takes recurrence relation coefficients
    n = len(an)

    # Step 1: Compute diags of jacobi 
    alpha = np.array([-bn[i] / an[i] for i in range(n)])

    # Step 2: Compute off diag of jacobi
    beta = np.array([np.sqrt(cn[i+1] / (an[i] * an[i+1])) for i in range(n - 1)])

    # Step 3: build jacobi
    J = np.zeros((n, n))
    np.fill_diagonal(J, alpha)
    for i in range(n - 1):
        J[i, i + 1] = beta[i]
        J[i + 1, i] = beta[i]
    print(J)
    # Step 4: QR Algorithm
    max_iter = 1000
    tol = 1e-12
    J = QR(J)

    # Step 5: Roots are the eigenvalues (diagonal of converged J)
    return np.sort(np.diag(J))


def generate_tchebycoeffs(n): #generates array of length n with tcheby 3 term coeffs
    an = np.ones(n)*2
    an[0] = 1
    bn = np.zeros(n)
    cn = np.ones(n)
    return an, bn, cn

def QR(A, max_iter, tol):
    max_iter = 1000
    tol = 1e-12
    for _ in range(max_iter):
        Q, R = np.linalg.qr(J)
        J_new = R @ Q
        if np.allclose(J, J_new, atol=tol):
            break
        J = J_new
    return J


driver()
