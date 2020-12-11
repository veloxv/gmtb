import numpy as np
from ..compute_alpha import compute_alpha
from ..best_weighted_mean import best_weighted_mean
from gmtb.util import pdist
import networkx as nx


def get_pairs(dist_mat):
    # cost matrix: negative distance and Diagonal is infinity
    # cost_matrix = -dist_mat
    # np.fill_diagonal(cost_matrix,np.inf)
    # assignment = munkres(cost_matrix).nonzero()

    g = nx.from_numpy_matrix(dist_mat)
    pairs = nx.max_weight_matching(g)

    return pairs


# implementation of the linear recursive reconstruction method
# returns:
#   rec_obj - reconstructed object
#   best_value - value of the reconstructed object (at the moment SOD)
def linear_recursive(object_set, weights, dist, kernel_func, dist_func, weighted_mean_func, kernel_matrix,
                     verbose=False):
    # return if only one object
    if len(object_set) == 1:
        return object_set[0], 0

    # save original set
    orig_set = object_set.copy()
    object_set = object_set.copy()

    # arrays for intermediate kernel values
    k_mean_set = kernel_matrix
    k_mean_mean = np.diag(kernel_matrix)

    # first best value: set median
    ind = np.argsort(dist)
    rec_obj = object_set[ind[0]]
    # best_value = np.sum(pdist(set1=orig_set,set2=[rec_obj],func=dist_func))
    best_value = np.sum(np.sqrt(k_mean_mean[ind[0]] - k_mean_set[ind[0], :]
                                - np.conjugate(k_mean_set[ind[0], :]) + np.diag(kernel_matrix)))

    # precompute k_median
    k_median_vec = np.sum(np.outer(weights, weights) * kernel_matrix) / np.sum(weights) ** 2

    while len(object_set) > 1:

        # sort by distance
        ind = np.argsort(dist)
        pairs = [x for x in zip(*[iter(ind)] * 2)]
        leftover = []

        if len(ind) % 2 == 1:
            leftover = [ind[-1]]

        # sort by pairs (see ewm)
        # dist_mat = pdist(object_set, dist_func)

        # prevent no assignment for zero elements
        # dist_mat = (dist_mat + 1) - np.eye(len(object_set))
        # pairs = get_pairs(dist_mat)
        # leftover = set(range(len(object_set))).difference([item for t in pairs for item in t])

        new_object_set = []
        new_k_mean_set = []
        new_k_mean_mean = []
        new_dist = []

        for (x, y) in pairs:

            # compute new weighted mean
            alpha = compute_alpha(weights, k_mean_set[x], k_mean_mean[x], k_mean_set[y],
                                  k_mean_mean[y], kernel_func(object_set[x], object_set[y]))

            nm, kms, kmm, bv = \
                best_weighted_mean(object_set[x], k_mean_set[x], k_mean_mean[x],
                                   object_set[y], k_mean_set[y], k_mean_mean[y],
                                   alpha, orig_set, kernel_matrix,
                                   kernel_func, dist_func,
                                   weighted_mean_func)

            new_object_set.append(nm)
            new_k_mean_set.append(kms)
            new_k_mean_mean.append(kmm)
            tmp_dist = k_median_vec \
                       - np.sum(weights * kms) / np.sum(weights) \
                       - np.conjugate(np.sum(weights * kms) / np.sum(weights)) \
                       + kmm

            new_dist.append(tmp_dist)

            # save best
            if bv < best_value:
                best_value = bv
                rec_obj = nm

        # if leftover objects: just copy
        for x in leftover:
            new_object_set.append(object_set[x])
            new_k_mean_mean.append(k_mean_mean[x])
            new_k_mean_set.append(k_mean_set[x])
            new_dist.append(dist[x])

        # if np.remainder(len(object_set), 2) == 1:
        #    new_object_set.append(object_set[-1])
        #    new_k_mean_mean.append(k_mean_mean[-1])
        #    new_k_mean_set.append(k_mean_set[-1])
        #    new_dist.append(dist[-1])

        if (verbose):
            print("SOD: %f" % np.sum(pdist(set1=orig_set, set2=[rec_obj], func=dist_func)))

        # prepare next iteration
        object_set = new_object_set
        k_mean_set = new_k_mean_set
        k_mean_mean = new_k_mean_mean
        dist = new_dist

    # end: return best result
    return rec_obj, best_value
