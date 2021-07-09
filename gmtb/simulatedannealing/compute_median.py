from gmtb.util import pdist
import numpy as np
import copy
## Implementation of simulated annealing for the computation of the generalized median

## debug
import matplotlib.pyplot as plt

# the objective function to be optimized

def compute_median(dataset, dist_func, neighbor_func, num_iter=100, debug=False):

    # custom sum of distance func (maybe for normalizing)
    def sod_func(x):
        return np.sum(pdist(set1=dataset, set2=[x], func=dist_func))

    # temperature: from 100 to 0
    def temperature(r):
        return 100 * (1-r)

    def P(sod_m, sod_m_new, t):
        if t == 0:
            return 0.0
        else:
            return np.exp(-(sod_m_new - sod_m) / t)

    # set start value: the set median (copy because it will be changed)
    m = copy.copy(dataset[np.argmin(np.sum(pdist(dataset, func=dist_func), axis=0))])
    sod_m = sod_func(m)
    m_best = m
    sod_m_best = sod_m

    #debug
    if debug:
        iterations = [0]
        values = [sod_m]


    # the simulated annealing iteration
    for i in range(num_iter):
        t = temperature((i+1)/num_iter)
        m_new = neighbor_func(copy.copy(m),t)
        sod_m_new = sod_func(m_new)

        # debug
        if debug:
            print(f"Iteration {i : 3}, Temperature: {round(t) : 4}, SOD: {sod_m : 7}, SOD_new: {sod_m_new : 7}, P: {P(sod_m,sod_m_new,t) : .4}")

        if sod_m_new < sod_m or (P(sod_m,sod_m_new,t) * 0.01) > np.random.rand():
            m = m_new
            sod_m = sod_m_new

            #debug
            if debug:
                iterations.append(i)
                values.append(sod_m_new)

            #save best so far
            if sod_m_new < sod_m:
                m_best = m
                sod_m_best = sod_m

        # optional: restart if too bad?


    # debug
    if debug:
        plt.figure
        plt.plot(iterations, values)
        plt.show()

    return m_best, sod_m_best
