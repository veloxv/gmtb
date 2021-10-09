function g = gmpd_chi_squared2_grad_no_emb(p,set_p, weights)
%GMPD_CHI_SQUARED2_GRAD computes the gradient of the Chi^2 SOD at point alpha. This version assumes that p2 is the median.    n = length(p);  
  
    n = length(p);
    g = zeros(1,n);
    
    for k = 1:n
        if (p(k) > eps)
            g(k) = sum(weights .* (1 - (set_p(:,k).^2 ./ p(k).^2)));
        end
    end
         
    
end