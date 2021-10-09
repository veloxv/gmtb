import numpy as np


def analytical_solution(set_p, distance, weights):
    # GMPD_ANALYTICAL_SOLUTION Provides analytical solutions to the Generalized median of several distances.
    # If there is no analytical solution, None is returned.

    raise NotImplementedError("Testen!")

    if distance == 'mean':
        weights = weights / np.sum(weights)
        p = np.mean(set_p * weights[:, np.newaxis], axis=0)
        return p / np.sum(p)

    elif distance == 'kullback-leibler1':  # corresponds to product rule
        weights = weights / np.sum(weights)
        return np.prod(set_p ** weights[:, np.newaxis]) / np.sum(np.prod(set_p ** weights[:,np.newaxis],axis=1),axis=0)

    elif distance == 'kullback-leibler2':  # corresponds to mean
        return weights[:, np.newaxis] * set_p / np.sum(weights)

    elif distance == 'chi-squared1':  # corresponds to harmonic mean
        weights = weights / np.sum(weights)
        return np.sum(np.sum(weights[:, np.newaxis] / set_p, axis=1) ** -1, axis=0) ** -1 \
               * np.sum(weights[:, np.newaxis] / set_p, axis=1) ** -1

    elif distance == 'chi-squared2':  # corresponds to quadratic mean
        weights = weights / np.sum(weights)
        return np.sum(weights[:, np.newaxis] * (set_p ** 2)) ** (1/2) \
               * np.sum(np.sum(weights * (set_p ** 2), axis=1) ** (1/2), axis=0) ** (-1)

    else:
        return None

