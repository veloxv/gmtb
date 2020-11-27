import numpy as np
from gmtb.util import pdist


def best_weighted_mean(oa, ob, alpha, object_set, kernel_matrix, kernel_func, dist_func, weighted_mean_func,complex):

    n = len(object_set)

    # compute weighted median
    new_mean1 = weighted_mean_func(oa, ob, alpha)
    new_mean2 = weighted_mean_func(ob, oa, 1-alpha)

    # compute sets
    k_mean_set1 = np.zeros(n)
    k_mean_set2 = np.zeros(n)

    for i in range(n):
        k_mean_set1[i] = kernel_func(object_set[i], new_mean1)
        k_mean_set2[i] = kernel_func(object_set[i], new_mean2)

    k_mean_mean1 = kernel_func(new_mean1, new_mean1)
    k_mean_mean2 = kernel_func(new_mean2, new_mean2)

    # compute best value, ignore negative values
    best_value1 = np.sum(np.sqrt(k_mean_mean1 - k_mean_set1 - np.conjugate(k_mean_set1) + np.diag(kernel_matrix)))
    best_value2 = np.sum(np.sqrt(k_mean_mean2 - k_mean_set2 - np.conjugate(k_mean_set2) + np.diag(kernel_matrix)))

    # compute best value using dist_func
    #best_value1 = np.sum(pdist(set1=[new_mean1],set2=object_set,func=dist_func))
    #best_value2 = np.sum(pdist(set1=[new_mean2],set2=object_set,func=dist_func))

    # return better result
    if best_value1 < best_value2:
        new_mean = new_mean1
        k_mean_set = k_mean_set1
        k_mean_mean = k_mean_mean1
        best_value = best_value1
    else:
        new_mean = new_mean2
        k_mean_set = k_mean_set2
        k_mean_mean = k_mean_mean2
        best_value = best_value2

    # in case of complex alpha: test if oa was better. (the "sign" of alpha disappears)
    if complex:
        k_mean_set_oa = np.zeros(n)

        for i in range(n):
            k_mean_set_oa[i] = kernel_func(object_set[i], new_mean1)

        k_mean_mean_oa = kernel_func(oa, oa)

        # compute best value, ignore negative values
        best_value_oa = np.sum(np.sqrt(k_mean_mean_oa - k_mean_set_oa - np.conjugate(k_mean_set_oa) + np.diag(kernel_matrix)))

        if best_value_oa < best_value:
            new_mean = oa
            k_mean_set = k_mean_set_oa
            k_mean_mean = k_mean_mean_oa
            best_value = best_value_oa

    return new_mean, k_mean_set, k_mean_mean, best_value