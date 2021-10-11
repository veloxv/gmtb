function g = gmpd_kullback_leibler2_grad(alpha,set_p,weights)
%GMPD_KULLBACK_LEIBLER2_GRAD computes the gradient of GMPD_KULLBACK_LEIBLER2_SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            
            g(k) = g(k) - weights(p) * sum((set_p(p,:) .* (-1/n + (1:n==k))) ./ (1/n - sum(alpha/n) + [alpha,0]));
       end
    end
          
    
end