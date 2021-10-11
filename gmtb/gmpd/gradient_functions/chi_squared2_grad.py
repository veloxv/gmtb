function g = gmpd_chi_squared2_grad(alpha,set_p, weights)
%GMPD_CHI_SQUARED2_GRAD computes the gradient of the Chi^2 SOD at point alpha. This version assumes that p2 is the median.

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    idx = c > eps; % ~= 0
    %const = 10;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            cur_p = set_p(p,:);           
                        
            delta = -1/n + (1:n==k);
            g(k) = g(k) + weights(p) * sum( (1 - (cur_p(idx).^2 ./ c(idx).^2)) .* delta(idx));
            
        end
    end
          
    
end