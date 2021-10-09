function g = gmpd_hellinger_grad_no_emb(p,set_p,weights)
%GMPD_HELLINGER_GRAD computes the gradient of the Hellinger SOD at point alpha

    n = length(p);  
    g = zeros(1,n);
      
   

        
    for i = 1:n
        g(i) = -sum(weights .* set_p(:,i) ./ (4*sqrt(p(i).*set_p(:,i)) .* gmpd_hellinger_dist(p,set_p)));
    end
    
    % in case of division by 0 (hellinger distance = 0)
    g(isinf(g) | isnan(g)) = 0.0001;
          
    
end