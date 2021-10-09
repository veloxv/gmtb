function d = gmpd_bhattacharyya_dist( p1, p2 )
%GMPD_BHATTACHARYYA_DIST Bhattacharyya distance function for discrete probability distributions

    % D(p1,p2) = -log(BC(p1,p2))
    % BC(p1,p2) = sum_i sqrt(p1(i) * p2(i))

    % real() in case of rounding errors
    d = real(-log(sum(sqrt(p1.*p2))));
    

end

