function g = gmpd_total_variation_grad(alpha,set_p,weights)
%GMPD_TOTAL_VARIATION_GRAD computes the gradient of the Total Variation SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            %g(k) = g(k) +  1/2 * sum( sign([alpha,0] + (1 - sum(alpha))/n - set_p(p,:)) .* (-1/n + (1:n==k)));
            g(k) = g(k) - (weights(p)/2) .* sum((-1/n + (1:n==k)) .* sign(set_p(p,:) - (1/n - sum(alpha)/n + [alpha , 0])));
       end
    end
    
end