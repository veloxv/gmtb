function g = gmpd_lukaszyk_karmowski_grad(alpha,set_p,weights)
%GMPD_LUKASZYK_KARMOWSKI_GRAD computes the gradient of the Lukaszyk-Karmowski SOD at point alpha


    g = zeros(1,length(alpha));
    n = length(alpha)+1;

    % meshgrid version: much faster
    [i,j] = meshgrid(1:n,1:n);
    
    ind = abs(i(:) - j(:))';
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
                prod = (-1/n + (i(:)'==k)) .* set_p(p,j(:)');
                g(k) = g(k) +  weights(p) * sum(ind .* prod);
       end
    end
          
    
end