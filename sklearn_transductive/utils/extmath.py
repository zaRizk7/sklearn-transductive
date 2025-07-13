import numpy as np

__all__ = [
    "centering_kernel",
    "sort_eigencomponents",
    "remove_significant_negative_eigenvalues",
]


def centering_kernel(n, dtype=np.float64):
    I = np.eye(n, dtype=dtype)  # noqa: E741
    J = np.ones((n, n), dtype)
    return I - J / n


def sort_eigencomponents(eigenvalues, eigenvectors):
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors


def remove_significant_negative_eigenvalues(lambdas):
    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6

    lambdas = np.real(lambdas)
    max_eig = lambdas.max()

    significant_neg_eigvals_index = lambdas < -significant_neg_ratio * max_eig
    significant_neg_eigvals_index &= lambdas < -significant_neg_value

    lambdas[significant_neg_eigvals_index] = 0

    return lambdas
