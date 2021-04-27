import numpy as np
from gmtb.util import pdist, lower_bound
import networkx as nx
from scipy.optimize import fminbound


# helper function to form pairs
def ewm_get_pairs(dist_mat, combination_strategy):

    if combination_strategy == 'default':
        g = nx.from_numpy_matrix(dist_mat)
        pairs = nx.max_weight_matching(g)
        return pairs

    elif combination_strategy == 'random':
        n = dist_mat.shape[0]
        np.random.permutation(n-1)
        iter_obj = iter(np.random.permutation(n))
        pairs = zip(iter_obj, iter_obj)
        return pairs

    elif combination_strategy == 'nn-based':
        ind = np.argsort(np.sum(dist_mat, 0))  # ascending?
        iter_obj = iter(ind)
        pairs = zip(iter_obj, iter_obj)
        return pairs

    elif combination_strategy == 'set-median':
        ind = np.argsort(np.sum(dist_mat, 0))
        pairs = [(ind[0], ind[i]) for i in range(1, len(ind))]
        return pairs

    elif combination_strategy == 'lp-based':
        _, x = lower_bound(dist_mat, 'lp', None, True)
        x_mat = x[np.newaxis,:] + x[:,np.newaxis]
        g = nx.from_numpy_matrix(x_mat)
        pairs = nx.max_weight_matching(g)
        return pairs



def ewm(dataset, dist_func, weighted_mean_func, alpha_search=False, additional_objects=False,
        combination_strategy='default',verbose=False):
    # EWM calculates the generalized median of objects using an evolutionary weighted mean approach
    # Published in "Evolutionary Weighted Mean Based Framework for Generalized Median Computation with
    # Application to Strings"
    # by Lucas Franek and Xiaoyi Jiang
    #
    # Parameters:
    #   dataset              - Cell array of objects.
    #   dist_func            - Function handle @(o1,o2). Distance function between objects o1 and o2. Objects are
    #                          single cells from dataset
    #   weighted_mean_func   - Function handle @(o1,o2,alpha). Weighted mean function between objects. o1, o2 are
    #                          single cells, alpha is a float
    #                          between 0 and 1.
    #   alpha_search         - Uses function minimization to compute an optimal alpha instead of using several with
    #                          equal distance
    #   additiona_objects    - Add Objects to the starting set by random combination of objects with the weighted mean
    #                          for more choice in pairs
    #   combination_strategy - The chosen Combination strategy for forming pairs ('default', 'random', 'set-median',
    #                          'lp-based')
    #
    # Returns:
    #   best_object          - Resulting best object according to best_crit
    #   best_crit_value      - Sum of Distance of the best object

    # check if there is only one object
    if len(dataset) == 1:
        return dataset[0], 0

    # initialize sets. Copy to not change anything
    original_set = dataset.copy()
    dataset = dataset.copy()

    if additional_objects:
        n = len(dataset)
        for i in range(n):
            comb = np.random.choice(n, 2, False)
            alpha = np.random.rand() / 2 + 0.25
            dataset.append(weighted_mean_func(dataset[comb[0]], dataset[comb[1]], alpha))

    crit_values = np.sum(pdist(dataset,dist_func),axis=1).tolist()
    last_best_crit = np.min(crit_values)

    # ********************************* Parameters ********************************* #
    w = 3
    max_iter = 5
    #n_max = min(20,2*len(dataset))
    n_max = 10

    # ********************************* BEGIN ITERATION ********************************* #
    iteration = 0
    stop = False
    stop_again = False

    while iteration < max_iter and ~stop:

        iteration = iteration+1
        n = len(dataset)

        # compute Matrix of distances / similarities
        dist_mat = pdist(dataset,dist_func)

        # ********* iterate over pairs of objects *********
        for (x, y) in ewm_get_pairs(dist_mat, combination_strategy):

            if alpha_search:
                # find best alpha for the weighted mean using linear search
                def search_crit(alpha):
                    return np.sum(pdist(original_set, set2=[weighted_mean_func(dataset[x], dataset[y], alpha)],
                                        func=dist_func))

                alpha = fminbound(search_crit, 0, 1, xtol=last_best_crit*0.00001, maxfun=10,disp=0)#[alpha, ~] = fminbnd(search_crit,0,1,options);
                new_obj = weighted_mean_func(dataset[x],dataset[y],alpha)
                dataset.append(new_obj)
                crit_values.append(np.sum(pdist(set1=original_set,set2=[new_obj],func=dist_func)))

            # use equidistant alphas
            else:
                for step in range(1,w+1):
                    alpha = step/(1+w)
                    new_obj = weighted_mean_func(dataset[x], dataset[y], alpha)

                    dataset.append(new_obj)
                    crit_values.append(np.sum(pdist(set1=original_set,set2=[new_obj],func=dist_func)))

        ## ********* delete from result set *********

        idx = np.argsort(crit_values)
        max_length = min(len(dataset),n_max)

        dataset = [dataset[i] for i in idx[:max_length]]
        crit_values = [crit_values[i] for i in idx[:max_length]]


        ## stopping criterium: no change for 2 iterations
        if abs(last_best_crit - crit_values[0]) <= 1e-5:
            if stop_again:
                stop = True
            else:
                stop_again = True

        else:
            last_best_crit = crit_values[0]
            stop_again = False



    # ********************************* END ITERATION ********************************* #
    if verbose:
        print('EWM terminated after %d / %d iterations' % (iteration, max_iter))

    return dataset[0], crit_values[0]