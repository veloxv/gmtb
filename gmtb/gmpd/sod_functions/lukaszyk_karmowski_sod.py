function sod = gmpd_lukaszyk_karmowski_sod(p,set_p,weights)
%GMPD_LUKASZYK_KARMOWSKI_SOD computes the SOD of a given alpha for the Lukaszyk Karmowski metric. 
    
    n = length(p);
    sod = 0;

    % meshgrid version
    [i,j] = meshgrid(1:n,1:n);
    ind = abs(i(:) - j(:))';
    
    for p_i = 1:size(set_p,1)
        prod = p(i(:)) .* set_p(p_i,j(:));
        sod = sod + weights(p_i) * sum(ind .* prod); 
    end
end