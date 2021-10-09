import numpy as np

def set_median( set_p : np.ndarray, sod_func, weights=None):

    if weights is None:
        weights = np.ones((set_p.shape[0],0))

    best_result = sod_func(set_p[0,:],set_p,weights)
    best_index = 0

    for i in range(1,set_p.shape[0]):
        new_result = sod_func(set_p[i,:],set_p,weights)
        
        if new_result < best_result:
            best_result = new_result
            best_index = i

    set_median = set_p[best_index,:]
    sod = best_result

    return set_median, sod

