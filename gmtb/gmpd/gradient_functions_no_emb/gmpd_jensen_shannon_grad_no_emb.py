function g = gmpd_jensen_shannon_grad_no_emb(p,set_p, weights)
%GMPD_JENSEN_SHANNON_GRAD computes the gradient of the Jensen-Shannon SOD at point alpha

    g = zeros(1,length(p));
    
    for i = 1:length(p)
        
        g(i) = sum((weights./2).*log((2.*p(i))./(p(i) + set_p(:,i))));

    end
          
    
end