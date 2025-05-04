import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.linalg import eig
import math as math
import matplotlib.pyplot as plt

pi = math.pi
e = math.e

rho= lambda x:  1000 * np.ones_like(x) # density
w = 2*pi*100 # circular frequency
c = 10 # sound speed
H = 1.0
alpha = .1 # attenuation in dB per wavelength
k_squared = (w/c)**2 * (1+ 1j * (alpha/(40*pi*math.log(e,10))))**2 # complex wave number


p_func = lambda x: 1/rho(x)
q_func = lambda x: k_squared/rho(x)
w_func = lambda x: 1/rho(x)


def chebyshev_nodes(N):
    return np.cos(pi * np.arange(N + 1) / N)

def chebyshev_weight(x):
    weights = 1 / np.sqrt(1 - x**2)
    
    # Handle division by zero gracefully
    weights[np.isinf(weights)] = 1  # Replace infinities with 1, if necessary
    return weights

def chebyshev_basis(N, x):
    T = np.zeros((N + 1, len(x)))
    for n in range(N + 1):
        T[n] = Chebyshev.basis(n)(x)
    return T

def chebyshev_basis_derivatives(N, x, order=1):
    Td = np.zeros((N + 1, len(x)))
    for n in range(N + 1):
        Tn = Chebyshev.basis(n).deriv(m=order)
        Td[n] = Tn(x)
    return Td

def galekin_matrices(N, p_func, q_func, w_func):
    x = chebyshev_nodes(N)
    wq = np.pi / N * np.ones_like(x)
    wq[0] /= 2
    wq[-1] /= 2
    weight = chebyshev_weight(x)

    # Basis and derivatives
    T = chebyshev_basis(N, x)
    T1 = chebyshev_basis_derivatives(N, x, order=1)
    T2 = chebyshev_basis_derivatives(N, x, order=2)

    p = p_func(x)
    q = q_func(x)
    w = w_func(x)

    A = np.zeros((N + 1, N + 1), dtype=np.complex128)
    B = np.zeros((N + 1, N + 1), dtype=np.complex128)

    for i in range(N + 1):
        for j in range(N + 1):
            integrand_A = (p * T2[j] + q * T[j]) * T[i] * weight
            integrand_B = w * T[j] * T[i] * weight
            A[i, j] = np.sum(wq * integrand_A)
            B[i, j] = np.sum(wq * integrand_B)

    T_at_minus1 = np.array([Chebyshev.basis(n)(-1) for n in range(N + 1)])
    T_at_plus1  = np.array([Chebyshev.basis(n)(+1) for n in range(N + 1)])

    A[0, :] = T_at_minus1
    B[0, :] = 0
    A[-1, :] = T_at_plus1
    B[-1, :] = 0
    return A, B, x




# Solve
N = 30
A, B, x = galekin_matrices(N, q_func, p_func, w_func)
eigvals, eigvecs = eig(A, B)

# Sort
idx = np.argsort(eigvals.real)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Convert basis expansion to function at x
T = chebyshev_basis(N, x)
phi0 = eigvecs[:, 0].real @ T

# Plot


plt.plot(x, phi0)
plt.xlabel("x")
plt.ylabel("φ(x)")
plt.title("First Eigenfunction (Real Part)")
plt.grid(True)
plt.show()