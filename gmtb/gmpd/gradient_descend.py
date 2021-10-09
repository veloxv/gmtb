import numpy as np
from scipy.optimize import fminbound
from gmtb.gmpd import convert



def gradient_descend(alpha : np.ndarray,set_p,gradient_function,sod_function, weights, epsilon=1, beta=0.8, version=2):
    # GMPD_GRADIENT_DESCEND is a simple gradient descend algorithm to find the generalized median

    alpha_new = alpha

    stop_threshold = 1e-10
    max_iter = 100
    
    n = alpha.size + 1
    
    cont = True
    iter = 0

    
    if version == 1:
        raise NotImplementedError()
        # %% first version: stepwise with epsilon and beta
        # g_new = 0

        # while cont && iter < max_iter  % or similar small number
        #     alpha = alpha_new;
        #     g = g_new;
        #     g_new = gradient_function(alpha_new,set_p,weights);
        #     alpha_new = alpha_new - epsilon * g_new;
        #
        #     % check if the change is big enough
        #     cont = (abs(g_new * g_new') > stop_threshold && epsilon > stop_threshold);
        #
        #
        #     iter = iter + 1;
        #     %disp([num2str(iter) ': norm:' num2str(g_new * g_new') ', epsilon: ' num2str(epsilon)]);
        #
        #     % check if we ran out of bounds
        #     p = 1/n - sum(alpha_new)/n + [alpha_new, 0];
        #     if (any(p < 0))
        #         % Try again with smaller step size if we were out of bounds
        #         epsilon = beta * epsilon;
        #         alpha_new = alpha;
        #
        #         % else: backtracking
        #     elseif sod_function(gmpd_convert(alpha_new - epsilon * gradient_function(alpha_new,set_p,weights),false),set_p,weights) > ...
        #             sod_function(gmpd_convert(alpha_new,false),set_p,weights) - epsilon/2 * norm(gradient_function(alpha_new,set_p,weights))^2
        #         epsilon = beta * epsilon;
        #     end
        #
        # end
        #   disp(iter);
    
    else:
        # second version: use fminbnd and line search
        # alpha = fminbound(search_crit, 0, 1, xtol=1e-10, maxfun=100,maxIter=100??,disp=0)

        grad_old = np.Inf
        
        while cont and iter < max_iter:
            # alternative: min search!
            # calculate max
            grad = gradient_function(alpha,set_p,weights)

            if np.any(np.isnan(grad)) or np.all(grad == 0):
                 return alpha

            # calculate the maximum value for epsilon before running out of bounds
            p = convert(alpha,False)
            max_epsilon = p / (np.concatenate((grad,0)) - np.sum(grad)/n)
            max_epsilon = np.min(max_epsilon(max_epsilon >= 0))

            # search result
            if np.size > 0:
                min_fun = lambda eps : sod_function(convert(alpha - eps * grad,False), set_p, weights)
                # try
                eps = fminbound(min_fun, 0, max_epsilon, xtol=1e-10, maxfun=100,disp=0)
                # catch
                #    epsilon = max_epsilon/2;

            else:
                eps = 0

            alpha = alpha - eps * grad

            iter = iter + 1

            cont = np.abs(np.dot(grad, grad) - np.dot(grad_old, grad_old)) > stop_threshold and epsilon > stop_threshold
            grad_old = grad

        return alpha