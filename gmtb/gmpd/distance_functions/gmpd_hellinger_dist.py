function d = gmpd_hellinger_dist( p1, p2 )
%GMPD_HELLINGER_DIST Hellinger distance function for discrete probability distributions

    % D(p1,p2) = sqrt(1-BC(p1,p2))
    % BC(p1,p2) = sum_i sqrt(p1(i) * p2(i))

    % real() in case of rounding errors
    d = sqrt(1-sum(sqrt(p1.*p2),2));
    

end

