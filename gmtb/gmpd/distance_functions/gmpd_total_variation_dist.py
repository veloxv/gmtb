function d = gmpd_total_variation_dist( p1, p2 )
%GMPD_TOTAL_VARIATION_DIST Total Variation Distance function for discrete probability distributions

    % D(p1,p2) = 1/2 sum_i |p1(i)-p2(i)|

    % real() in case of rounding errors
    d = 1/2 * sum(abs(p1 - p2));
    

end

