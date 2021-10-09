function d = gmpd_chi_squared1_dist( p1, p2 )
%GMPD_CHI_SQUARED1_DIST Chi^2 distance function for discrete probability distributions

    % D(p1,p2) = sum_i (p1(i)-p2(i))^2/p2(i)

    % real() in case of rounding errors
    idx = p2 > eps; %~=0;
    d = sum((p1(idx) - p2(idx)).^2 ./ p2(idx));
    

end

