function g = gmpd_hellinger_grad(alpha,set_p,weights)
%GMPD_HELLINGER_GRAD computes the gradient of the Hellinger SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            cur_p = set_p(p,:);
            part1 = -sum(sqrt(cur_p) .* (-1/n + (1:n==k)) .* sqrt(c).^-1);
            part2 = sqrt(1-sum(sqrt(c .* cur_p)))^-1;

            % Prevent errors in part1 and part2 computation
            if(isnan(part1) || isinf(part1) || ~isreal(part1)); part1 = 0; end
            if(isnan(part2) || isinf(part2) || ~isreal(part2)); part2 = 0; end
            
            
            
            new_g = ( part1 .* part2 ) / 4;
            
            % prevent errors if the square-root was zero. 
            % this may cause the gradient to go in a slightly different direction
            if isinf(new_g); new_g = 0; end
            
            % real to prevent rounding errors
            g(k) = g(k) + weights(p) * real(new_g);
        end
    end
          
    
end