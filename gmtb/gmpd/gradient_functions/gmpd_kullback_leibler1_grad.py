function g = gmpd_kullback_leibler1_grad(alpha,set_p,weights)
%GMPD_KULLBACK_LEIBLER1_GRAD computes the gradient of GMPD_KULLBACK_LEIBLER_SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            cur_p = set_p(p,:);
            
            g(k) = g(k) + weights(p) * sum((-1/n+(1:n==k)) .* (log( (1/n - sum(alpha/n) + [alpha,0]) ./ cur_p ) + 1));
       end
    end
          
    
end