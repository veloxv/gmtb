import numpy as np

from gmtb.gmpd import set_median
from gmtb.gmpd.analytical_solution import analytical_solution
from gmtb.gmpd.init_alpha import init_alpha
from gmtb.gmpd.gradient_descend import gradient_descend
from gmtb.gmpd.convert import convert
from gmtb.gmpd.select_distance_function import select_distance_function


# GMPD Main Function for the Generalized Median of Probability Distributions
def gmpd(set_p : np.ndarray, distance, weights=None, use_gradient=False, epsilon=None, beta=None, version=2, norm_type='embedding'):

    _, grad_func, grad_func_no_emb, sod_func, epsilon_var, beta_var, prevent_zero = select_distance_function(distance)

    if epsilon is None:
        epsilon = epsilon_var

    if beta is None:
        beta = beta_var
    
    if weights is None:
        weights = np.ones((set_p.shape[0],1))

    # check if there is an analytical solution
    if not use_gradient:
        p = analytical_solution(set_p, distance, weights)

        if not p is None:
            if prevent_zero:
                sod = sod_func((p+0.000001 / np.sum(p+0.000001)), set_p + 0.000001 / np.sum(set_p+0.000001), weights)
            else:
                sod = sod_func(p,set_p,weights)

            return p, sod

    if prevent_zero:
        set_p[set_p < 0.0000001] = 0.0000001
        set_p = set_p / np.sum(set_p, axis=1)
    
    
    if norm_type == 'embedding':
        # Use Embedding for computation
        # find starting value (set median)
        alpha = init_alpha(set_p,sod_func,weights)

        # do gradient descend
        alpha_new = gradient_descend(alpha,set_p,grad_func,sod_func, weights, epsilon, beta, version)

        # convert back to probability distribution
        p = convert(alpha_new, False)

        # compute SOD
        sod = sod_func(p,set_p, weights)
        
    if norm_type == 'normalize_all':
        raise NotImplementedError()
        # p,_ = set_median(set_p,sod_func,weights)
        # p = gradient_descend_no_emb(p,set_p,grad_func_no_emb,sod_func, weights, epsilon, beta, version, lambda p : p / np.sum(p))
        # sod = sod_func(p,set_p,weights)

    elif norm_type == 'normalize_once':
        raise NotImplementedError()
        # p,_ = gmpd_set_median(set_p,sod_func,weights)
        # p = gmpd_gradient_descend_no_emb(p,set_p,grad_func_no_emb,sod_func, weights, epsilon, beta, version, lambda p : p)
        # p = p / np.sum(p)
        # sod = sod_func(p,set_p,weights)

    elif norm_type == 'simplex_all':
        raise NotImplementedError()
        # p,_ = set_median(set_p,sod_func,weights)
        # p = gradient_descend_no_emb(p,set_p,grad_func_no_emb,sod_func, weights, epsilon, beta, version, @gmpd_simplex_proj)
        # sod = sod_func(p,set_p,weights)

    elif norm_type == 'simplex_once':
        raise NotImplementedError()
        # p,_ = set_median(set_p,sod_func,weights)
        # p = gradient_descend_no_emb(p,set_p,grad_func_no_emb,sod_func, weights, epsilon, beta, version, lambda p : p)
        # p = simplex_proj(p)
        # sod = sod_func(p,set_p,weights)

    else:
        raise AttributeError("norm_type must be 'normalize_all', 'normalize_once', 'simplex_all' or 'simplex_once'")
    return p, sod
