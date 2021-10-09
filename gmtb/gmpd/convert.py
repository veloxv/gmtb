import numpy as np


def convert( in_array : np.ndarray, to_alpha=True):
    if to_alpha:
        raise NotImplementedError("war: end-1, so richtig?")
        return in_array[:,:-1] - in_array[:,-1]
    else:
        return np.concatenate((in_array, 0)) + (1 - np.sum(in_array,axis=1)) / (in_array.shape[1]+1)
