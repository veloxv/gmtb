function [p_new] = gmpd_gradient_descend_no_emb(p,set_p,gradient_function,sod_function, weights, epsilon, beta, version, normalize_func)
%GMPD_GRADIENT_DESCEND is a simple gradient descend algorithm to find the generalized median

% TODO: parameters?
    p_new = p; % a number different from X
    
    if (nargin < 8)
        version = 2;  
    end
    
    if (nargin < 7)
        beta = 0.8;
    end
    
    if (nargin < 6)
        epsilon = 1;  
    end
    
    if (nargin < 9)
        normalize_func = @(p) p ./ sum(p);
    end
    
    beta = 0.8;
    epsilon = 1;
    

    stop_threshold = 1e-10;
    max_iter = 100;
    
    n = length(p)+1;
    
    cont = true;
    iter = 0;
    g_new = 0;
    
    if version == 1
        %% first version: stepwise with epsilon and beta
        while cont && iter < max_iter  % or similar small number
            p = p_new;
            g = g_new;
            g_new = gradient_function(p_new,set_p,weights);
            p_new = p_new - epsilon * g_new;
            
            % normalization
            p_new = normalize_func(p_new);

            % check if the change is big enough
            cont = (abs(g_new * g_new') > stop_threshold && epsilon > stop_threshold);


            iter = iter + 1;
            %disp([num2str(iter) ': norm:' num2str(g_new * g_new') ', epsilon: ' num2str(epsilon)]);

            % check if we ran out of bounds
            if (any(p_new < 0))
                % Try again with smaller step size if we were out of bounds
                epsilon = beta * epsilon;
                p_new = p;

                % else: backtracking
            elseif sod_function(p_new - epsilon * gradient_function(p_new,set_p,weights),set_p,weights) > ...
                    sod_function(p_new,set_p,weights) - epsilon/2 * norm(gradient_function(p_new,set_p,weights))^2
                epsilon = beta * epsilon;
            end

        end
%         disp(iter);
    
    else
        %% second version: use fminbnd and line search

%         line_search_opts = ...
%         struct('Display','off', 'MaxFunEvals',100, 'MaxIter',100, 'TolX',1e-10);
       
        grad_old = Inf;
        
        while cont && iter < max_iter
            % alternative: min search!
            % calculate max 
             grad = gradient_function(p,set_p,weights);

             if (any(isnan(grad)) || all(grad == 0))
                 p_new = p;
                 return
             end

            % calculate the maximum value for epsilon before running out of bounds
            max_epsilon = [p ./ grad, 
                           (p-1) ./ grad];        
            max_epsilon = min(max_epsilon(max_epsilon >= 0));

            % search result
            if (~isempty(max_epsilon))
                min_fun = @(epsilon) sod_function(p - epsilon * grad,set_p,weights);
                epsilon = fminbnd(min_fun,0,max_epsilon);%,line_search_opts);
            else
                epsilon = 0;
            end
            p = p - epsilon * grad;
            
            % normalize to get true probability function 
            p = normalize_func(p);

        

%             line_search_opts.TolX = line_search_opts.TolX/10;
            iter = iter + 1;
            
            %stop_threshold;
            cont = abs(grad * grad' - grad_old * grad_old') > 1e-4  && epsilon > 1e-4;
            grad_old = grad;
        end
%         disp(iter);

        p_new = p;
    end    
end