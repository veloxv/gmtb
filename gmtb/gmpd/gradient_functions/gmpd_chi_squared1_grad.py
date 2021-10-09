function g = gmpd_chi_squared1_grad(alpha,set_p,weights)
%GMPD_CHI_SQUARED1_GRAD computes the gradient of the Chi^2 SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            cur_p = set_p(p,:);
            idx = cur_p > eps; % ~= 0;
            
            part1 = (-1/n + (1:n==k));
            part2 = 2 * (c-cur_p);
            
            g(k) = g(k) + weights(p) * sum( (part1(idx) .* part2(idx)) ./ cur_p(idx));
        end
    end
          
    
end