function d = gmpd_separation1_dist( p1, p2 )
%GMPD_SEPARATION1_DIST Separation distance function for discrete probability distributions

    % D(p1,p2) = max_i ( 1 - p1(i)/p2(i))

    d = max(1-(p1./p2));
    

end

