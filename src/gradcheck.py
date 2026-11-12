# finite-difference gradient checking
import numpy as np


# central finite differences, eq. (13)
def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus, x_minus = x.copy(), x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        g[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return g

def rel_error(analytic, numeric):
    # largest absolute difference, relative to the largest gradient entry
    return np.abs(analytic - numeric).max() / (np.abs(numeric).max() + 1e-12)
