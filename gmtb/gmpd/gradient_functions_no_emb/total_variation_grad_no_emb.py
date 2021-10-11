function g = gmpd_total_variation_grad_no_emb(p,set_p,weights)
%GMPD_TOTAL_VARIATION_GRAD computes the gradient of the Total Variation SOD at point alpha

    n = length(p);
    g = zeros(1,n);
    
    
    for i = 1:length(p)
        
        g(i) = sum(weights./2 .* sign(p(i) - set_p(:,i)));

    end
    
end