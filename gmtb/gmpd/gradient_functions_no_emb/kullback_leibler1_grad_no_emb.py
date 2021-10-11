function g = gmpd_kullback_leibler1_grad_no_emb(p,set_p,weights)
%GMPD_KULLBACK_LEIBLER1_GRAD computes the gradient of GMPD_KULLBACK_LEIBLER_SOD at point alpha
    
    n = length(p);  
    g = zeros(1,n);

    for k = 1:n
        g(k) = sum(weights .* (1+ log(p(k)./set_p(:,k))));
    end
          
    
end