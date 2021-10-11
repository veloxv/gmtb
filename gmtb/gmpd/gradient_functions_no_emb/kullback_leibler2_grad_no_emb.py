function g = gmpd_kullback_leibler2_grad_no_emb(p,set_p,weights)
%GMPD_KULLBACK_LEIBLER2_GRAD computes the gradient of GMPD_KULLBACK_LEIBLER2_SOD at point alpha

    n = length(p);  
    g = zeros(1,n);

    for k = 1:n
        g(k) = sum(weights .* (- set_p(:,k)/p(k)));
    end         
    
end