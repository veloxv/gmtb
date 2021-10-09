function g = gmpd_lukaszyk_karmowski_grad_no_emb(p,set_p,weights)
%GMPD_LUKASZYK_KARMOWSKI_GRAD computes the gradient of the Lukaszyk-Karmowski SOD at point alpha

    n = length(p);
    g = zeros(1,length(p));    


    for i = 1:length(p)
        ind = abs(i-(1:n));
        g(i) = sum(weights .* sum(ind.*set_p,2));
    end
          
    
end