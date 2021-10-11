function d = gmpd_jensen_shannon_dist( p1, p2 )
%GMPD_JENSEN_SHANNON_DIST Jensen-Shannon divergence function for discrete probability distributions

    % D(p1,p2) = -log(BC(p1,p2))
    % BC(p1,p2) = sum_i sqrt(p1(i) * p2(i))

    % real() in case of rounding errors
    pM = (p1+p2)/2;
    d = 1/2 * (gmpd_kullback_leibler1_dist(p1,pM) + gmpd_kullback_leibler1_dist(p2,pM));
    

end

