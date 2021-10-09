function d = gmpd_kullback_leibler1_dist( p1, p2 )
%GMPD_KULLBACK_LEIBLER1_DIST KL-Divergence function for discrete probability distributions

    % D(p1,p2) = sum_i p1(i) log(p1(i)/p2(i))

    d = sum(p1 .* real(log(p1./p2)));
    

end

