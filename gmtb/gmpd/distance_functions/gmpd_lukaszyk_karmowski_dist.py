function d = gmpd_lukaszyk_karmowski_dist( p1, p2 )
%GMPD_LUKASZYK_KARMOWSKI_DIST Lukaszyk-Karmowski metric function for discrete probability distributions

    % D(p1,p2) = sum_i sum_j |i-j| p1(i) p2(j)

    % real() in case of rounding errors
       
    n = length(p1);
    [i,j] = meshgrid(1:n,1:n);
    ind = abs(i(:) - j(:));
    prod = p1(i(:)) .* p2(j(:));
    d = sum(ind(:) .* prod(:)); 

end

