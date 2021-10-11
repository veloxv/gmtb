function d = gmpd_kullback_leibler2_dist( p1, p2 )
%GMPD_KULLBACK_LEIBLER2_DIST KL-Divergence function for discrete probability distributions. This version reverses p1 and p2

    % D(p1,p2) = sum_i p1(i) log(p1(i)/p2(i))

    d = sum(p2 .* log(p2./p1));
    

end

