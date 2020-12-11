import numpy as np
from gmtb.util import pdist


def best_weighted_mean(o_a, k_oa_set, k_oa_oa, o_b, k_ob_set, k_ob_ob,
                       alpha, object_set, kernel_matrix, kernel_func,
                       dist_func, weighted_mean_func):

    n = len(object_set)

    # compute weighted median
    new_mean_1 = weighted_mean_func(o_a, o_b, alpha)
    new_mean_2 = weighted_mean_func(o_b, o_a, 1-alpha)

    # compute sets
    k_mean_set_1 = np.zeros(n)
    k_mean_set_2 = np.zeros(n)

    for i in range(n):
        k_mean_set_1[i] = kernel_func(object_set[i], new_mean_1)
        k_mean_set_2[i] = kernel_func(object_set[i], new_mean_2)

    k_mean_mean_1 = kernel_func(new_mean_1, new_mean_1)
    k_mean_mean_2 = kernel_func(new_mean_2, new_mean_2)

    # compute best value, ignore negative values
    best_value_1 = np.sum(np.sqrt(k_mean_mean_1 - k_mean_set_1 - np.conjugate(k_mean_set_1) + np.diag(kernel_matrix)))
    best_value_2 = np.sum(np.sqrt(k_mean_mean_2 - k_mean_set_2 - np.conjugate(k_mean_set_2) + np.diag(kernel_matrix)))
    best_value_a = np.sum(np.sqrt(k_oa_oa - k_oa_set - np.conjugate(k_oa_set) + np.diag(kernel_matrix)))
    best_value_b = np.sum(np.sqrt(k_ob_ob - k_ob_set - np.conjugate(k_ob_set) + np.diag(kernel_matrix)))

    arg_min = np.argmin(np.absolute([best_value_1, best_value_2, best_value_a, best_value_b]))

    new_mean = None
    k_mean_set = None
    k_mean_mean = None
    best_value = None

    if arg_min == 0:
        new_mean = new_mean_1
        k_mean_set = k_mean_set_1
        k_mean_mean = k_mean_mean_1
        best_value = best_value_1
    elif arg_min == 1:
        new_mean = new_mean_2
        k_mean_set = k_mean_set_2
        k_mean_mean = k_mean_mean_2
        best_value = best_value_2
    elif arg_min == 2:
        new_mean = o_a
        k_mean_set = k_oa_set
        k_mean_mean = k_oa_oa
        best_value = best_value_a
    elif arg_min == 3:
        new_mean = o_b
        k_mean_set = k_ob_set
        k_mean_mean = k_ob_ob
        best_value = best_value_b

    return new_mean, k_mean_set, k_mean_mean, best_value
