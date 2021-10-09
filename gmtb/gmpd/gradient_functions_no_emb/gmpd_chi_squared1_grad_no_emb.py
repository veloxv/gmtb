function g = gmpd_chi_squared1_grad_no_emb(p,set_p,weights)
%GMPD_CHI_SQUARED1_GRAD computes the gradient of the Chi^2 SOD at point p
    
    n = length(p);  
    g = zeros(1,n);
    
    for k = 1:n
        idx = (set_p(:,k) > 0);
        g(k) = sum(weights(idx) .* 2 .* (p(k) - set_p(idx,k))./set_p(idx,k));
    end
       
    
end