function d = gmpd_maximum_mean_dist( p1, p2 )
%GMPD_MAXIMUM_MEAN_DIST Maximum Mean discrepancy function for discrete probability distributions

    % D(p1,p2) = (sum_i i (p1(i) - p2(i)))^2

    % real() in case of rounding errors
    d = sum((1:length(p1))' .* (p1-p2))^2;
    

end

