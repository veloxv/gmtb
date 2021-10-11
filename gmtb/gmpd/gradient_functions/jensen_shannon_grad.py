function g = gmpd_jensen_shannon_grad(alpha,set_p, weights)
%GMPD_JENSEN_SHANNON_GRAD computes the gradient of the Jensen-Shannon SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    p = [alpha,0] + (1 - sum(alpha))/n;
    
    for k = 1:length(alpha)
        for p_i = 1:size(set_p,1)
            cur_p = set_p(p_i,:);
            
            %part1 = - (cur_p .* (-1/n + (1:n==k))) ./ (cur_p .* p);
            %part2 = ((-1/n + (1:n==k)) .* ( (cur_p + p) .* log( (2*p) ./ (cur_p + p)) + cur_p)) ./ (cur_p + p);
            

            %g(k) = g(k) + weights(p_i)/2 * sum(part1 + part2);
            
            g(k) = g(k) + weights(p_i)/2 * sum( (-1/n + (1:n==k)) .* log ( (2 * p) ./ (p + cur_p)) );
        end
    end
          
    
end