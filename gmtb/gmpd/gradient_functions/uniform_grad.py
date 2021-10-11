function g = gmpd_uniform_grad(alpha,set_p,weights)
%GMPD_UNIFORM_GRAD computes the gradient of the Kolmogorov or Uniform metric at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            
            cur_p = set_p(p,:);
            
            % max index
            [~, m] = max(abs(c-cur_p));
            
            %g(k) = g(k) + weights(p) * (-1/n  + (k==m)) .*  (-sign(c - cur_p));
            g(k) = g(k) + weights(p) * (-1/n  + (k==m)) .*  (-sign(cur_p(m) - c(m)));
        end
    end
         
    
end