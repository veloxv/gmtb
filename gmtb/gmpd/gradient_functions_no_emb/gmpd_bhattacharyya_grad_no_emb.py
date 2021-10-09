function g = gmpd_bhattacharyya_grad_no_emb(p,set_p,weights)
%GMPD_BHATTACHARYYA_GRAD computes the gradient of the Bhattacharyya SOD at point alpha
 
    n = length(p);    
    g = zeros(1,n);
   
    
    for i = 1:n
        for pi = 1:size(set_p,1)
            cur_p = set_p(pi,:);
            
            new_g = weights(pi) * cur_p(i) / (2 * sqrt(p(i)*cur_p(i))*sum(sqrt(p.*cur_p)));
            
            % real to prevent rounding errors
            g(i) = g(i) + real(new_g);
        end
    end
          
    
end