function d = gmpd_chi_squared2_dist( p1, p2 )
%GMPD_CHI_SQUARED2_DIST Chi^2 distance function for discrete probability distributions. This version reverses p1 and p2

    % D(p1,p2) = sum_i (p1(i)-p2(i))^2/p2(i)

    % real() in case of rounding errors
    idx = p1 > eps; %~=0;
    d = sum((p2(idx) - p1(idx)).^2 ./ p1(idx));
    

end

