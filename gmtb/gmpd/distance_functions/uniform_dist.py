function d = gmpd_uniform_dist( p1, p2 )
%GMPD_UNIFORM_DIST Kolmogorov or Uniform metric for discrete probability distributions

    % D(p1,p2) = sup_i | p1(i)  - p2(i) |

    % sup becomes max in our case
    d = max(p1-p2);
    

end

