import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

# Define number of basis functions
N = 5

# Define weight function for Chebyshev polynomials
def w(x):
    return 1 / np.sqrt(1 - x**2)

# Define f(x)
def f(x):
    return np.pi**2 * np.sin(np.pi * x)

# Generate Chebyshev polynomials of first kind
def T(n, x):
    return np.cos(n * np.arccos(x))

# Modified basis functions: φ_n(x) = T_n(x) - T_{n+2}(x)
def phi(n, x):
    return T(n, x) - T(n + 2, x)

# Second derivative of Chebyshev polynomials using recursion
def T_dd(n, x):
    if n < 2:
        return np.zeros_like(x)
    else:
        # Derivative using recurrence (or use symbolic differentiation for precision)
        # Approximation via finite differences for demo purposes
        h = 1e-5
        return (T(n, x + h) - 2 * T(n, x) + T(n, x - h)) / h**2

# Second derivative of modified basis
def phi_dd(n, x):
    return T_dd(n, x) - T_dd(n + 2, x)

# Compute Galerkin matrix A and RHS vector b
A = np.zeros((N, N))
b = np.zeros(N)

# Use Gaussian-Chebyshev quadrature points and weights
quad_N = 200
xq = np.cos(np.pi * (np.arange(1, quad_N + 1) - 0.5) / quad_N)
wq = np.pi / quad_N  # All weights are equal in Gauss-Chebyshev

for m in range(N):
    for n in range(N):
        integrand = phi_dd(n + 1, xq) * phi(m + 1, xq)
        A[m, n] = np.sum(integrand * wq)
    integrand_b = f(xq) * phi(m + 1, xq)
    b[m] = np.sum(integrand_b * wq)

# Solve the linear system
a = np.linalg.solve(A, -b)

# Define approximate solution
def u_approx(x):
    return sum(a[n] * phi(n + 1, x) for n in range(N))

# Plot the result
x_plot = np.linspace(-1, 1, 500)
u_plot = u_approx(x_plot)
u_exact = np.sin(np.pi * x_plot)

plt.plot(x_plot, u_plot, label='Galerkin Approximation', lw=2)
plt.plot(x_plot, u_exact, '--', label='Exact Solution', lw=2)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Chebyshev-Galerkin Approximation')
plt.grid(True)
plt.show()


# Error convergence study
N_values = np.arange(2, 21)
errors_L2 = []

x_eval = np.linspace(-1, 1, 1000)
u_exact_eval = np.sin(np.pi * x_eval)

for N in N_values:
    # Recompute everything for each N
    A = np.zeros((N, N))
    b = np.zeros(N)
    for m in range(N):
        for n in range(N):
            integrand = phi_dd(n + 1, xq) * phi(m + 1, xq)
            A[m, n] = np.sum(integrand * wq)
        integrand_b = f(xq) * phi(m + 1, xq)
        b[m] = np.sum(integrand_b * wq)
    a = np.linalg.solve(A, -b)

    def u_approx_current(x):
        return sum(a[n] * phi(n + 1, x) for n in range(N))

    u_approx_eval = u_approx_current(x_eval)
    error_L2 = np.sqrt(np.sum((u_approx_eval - u_exact_eval)**2) * (2 / len(x_eval)))
    errors_L2.append(error_L2)

# Plot convergence
plt.figure()
plt.plot(N_values, errors_L2, 'o-', label='$L^2$ Error')
plt.yscale('log')
plt.xlabel('Number of Basis Functions (N)')
plt.ylabel('Error')
plt.title('Error Convergence of Chebyshev-Galerkin Method')
plt.legend()
plt.grid(True)
plt.show()
