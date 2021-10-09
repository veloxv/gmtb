function d = gmpd_separation2_dist( p1, p2 )
%GMPD_SEPARATION1_DIST Separation distance function for discrete probability distributions, reverse version

    % D(p1,p2) = max_i ( 1 - p1(i)/p2(i))

    % real() in case of rounding errors
    d = max(1-(p2./p1));
    

end

