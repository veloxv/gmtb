import numpy as np


def median_weights_iter(n, kernel_matrix, start, normalize):
    # return values
    weights_median_set = np.full(n, 1e10, dtype=complex)
    weights_median_set_new = start
    k_median_set = np.zeros(n, dtype=complex)
    k_median_median = 0

    # stop conditions
    nan_stop = False
    threshold = 1e-10
    max_iter = 1000
    iter = 0
    first_complex = np.nan

    while np.sum(np.abs(weights_median_set - weights_median_set_new)) > threshold and iter < max_iter:

        weights_median_set = weights_median_set_new
        #k_median_median = k_median_median_new
        #k_median_set = k_median_set_new

        # kernel between median and median
        k_median_median = np.sum(kernel_matrix * np.outer(weights_median_set, np.conjugate(weights_median_set))) / \
                              np.power(np.absolute(np.sum(weights_median_set)), 2)

        # kernel between objects and median
        for i in range(n):
            k_median_set[i] = np.sum(kernel_matrix[:, i] * weights_median_set) / np.sum(weights_median_set)

        # compute weights
        tmp_weights = np.diag(kernel_matrix) - k_median_set - np.conjugate(k_median_set) + k_median_median

        # if any tmp_weight is 0: that one must be the median (infinite weight)
        if np.any(tmp_weights == 0):
            ind = tmp_weights == 0
            k_median_set = kernel_matrix[ind, :]
            k_median_median = kernel_matrix[ind, ind]
            weights_median_set_new = np.zeros_like(weights_median_set_new)
            weights_median_set_new[ind] = 1
            return weights_median_set_new, k_median_set, k_median_median, nan_stop, iter, max_iter, first_complex

        # else: compute weights normally, but use absolute to ensure real positive weights
        weights_median_set_new = (1 / np.sqrt(tmp_weights))

        if np.isnan(first_complex) and np.any(np.iscomplex(weights_median_set_new)):
            first_complex = iter

        iter = iter + 1

    #if not np.isnan(first_complex):
    #    print(weights_median_set_new[np.logical_and(np.iscomplex(weights_median_set_new),np.abs(np.imag(weights_median_set_new)) > 1e-18)])

    if np.any(np.isnan(weights_median_set_new)):
        nan_stop = True

    return weights_median_set_new, k_median_set, k_median_median, nan_stop, iter, max_iter, first_complex


def compute_weights(kernel_matrix, normalize=False):
    n = np.shape(kernel_matrix)[0]

    start = np.ones(n, dtype=complex)
    # start = np.sum(kernel_matrix).astype(complex) / np.power(n, 2) - (2 / n * np.sum(kernel_matrix, 1)) + np.diag(kernel_matrix)
    # start[start < 1e-5] = 1e-5
    start = np.sqrt(start)

    max_iter = 5
    iter = 0
    weights_median_set = None
    k_median_set = None
    k_median_median = None
    nan_stop = True

    # as long as iteration not stopped and there was no stop because of negative values
    while iter < max_iter and nan_stop:
        weights_median_set, k_median_set, k_median_median, nan_stop, iter2, max_iter2, first_complex = \
            median_weights_iter(n, kernel_matrix, start, normalize)

        start = np.random.rand(n) + 1
        start = start.astype(complex)
        iter = iter + 1

    # clean up weights: only real OR imaginary should be left, rest is rounding errors
    tol=1e-10
    weights_median_set.real[abs(weights_median_set.real) < tol] = 0.0
    weights_median_set.imag[abs(weights_median_set.imag) < tol] = 0.0

    return weights_median_set, k_median_set, k_median_median#, iter2, max_iter2, first_complex
