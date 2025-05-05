import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def chebyshev_differentiation_matrix(N):
    """
    Construct the Chebyshev differentiation matrix D and the Chebyshev grid x.
    """
    if N == 0:
        return np.array([[0]]), np.array([1.0])
    
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c = c * (-1)**np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T + np.eye(N + 1)  # Avoid division by zero
    D = (c[:, None] / c[None, :]) / dX
    D -= np.diag(np.sum(D, axis=1))
    return D, x

def solve_sturm_liouville_chebyshev(N):
    """
    Solves y'' + λy = 0 with y(0)=y(1)=0 using Chebyshev spectral method.
    Returns first four eigenvalues and eigenfunctions.
    """
    D, x_cheb = chebyshev_differentiation_matrix(N)
    D2 = np.dot(D, D)

    # Remove boundary rows and columns to apply Dirichlet BCs
    D2_inner = D2[1:-1, 1:-1]
    x_inner = x_cheb[1:-1]

    # Solve eigenvalue problem -D2 y = λ y
    eigvals, eigvecs = eig(-D2_inner)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # Sort eigenvalues and corresponding eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs, x_cheb

from scipy.interpolate import interp1d

def plot_eigenfunctions(eigvecs, x_cheb, num_modes=10):
    """
    Plot the first `num_modes` eigenfunctions smoothly using interpolation.
    """
    N = len(x_cheb) - 1
    x_plot = (x_cheb + 1) / 2  # Map Chebyshev points from [-1,1] to [0,1]
    x_dense = np.linspace(0, 1, 500)  # Dense uniform grid for smooth plots

    plt.figure(figsize=(10, 6))
    for k in range(8, num_modes):
        yk = np.zeros(N + 1)
        yk[1:-1] = eigvecs[:, k]  # Apply boundary conditions (0 at ends)

        # Interpolate eigenfunction on the dense grid
        interpolator = interp1d(x_plot, yk, kind='cubic')
        yk_dense = interpolator(x_dense)

        # Exact solution for comparison
        y_exact = np.sin((k + 1) * np.pi * x_dense)

        plt.plot(x_dense, yk_dense, label=f"$\\tilde{{y}}_{k+1}(x)$ (approx)")
        plt.plot(x_dense, y_exact, '--', label=f"$y_{k+1}(x)$ (exact)")

    plt.title("First Eigenfunctions (Chebyshev Spectral Method)", fontsize=14)
    plt.xlabel("$x$")
    plt.ylabel("$y(x)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_eigenvalue_comparison(eigvals, num_modes=4):
    """
    Print the first `num_modes` numerical and exact eigenvalues.
    """
    print("First four eigenvalues (Chebyshev spectral method):")
    print(f"{'Mode':>5} {'Computed λ':>15} {'Exact λ = (kπ)^2':>20} {'Abs Error':>15}")
    for k in range(num_modes):
        computed = eigvals[k]
        exact = (np.pi * (k + 1)) ** 2
        error = abs(computed - exact)
        print(f"{k+1:5d} {computed:15.8f} {exact:20.8f} {error:15.2e}")

# === MAIN EXECUTION ===
N = 50  # Chebyshev points
eigvals, eigvecs, x_cheb = solve_sturm_liouville_chebyshev(N)
print_eigenvalue_comparison(eigvals)
plot_eigenfunctions(eigvecs, x_cheb)

plt.plot()
