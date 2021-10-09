function g = gmpd_bhattacharyya_grad(alpha,set_p,weights)
%GMPD_BHATTACHARYYA_GRAD computes the gradient of the Bhattacharyya SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            cur_p = set_p(p,:);
            
            part1 = sum(((-1/n + (1:n==k)).* cur_p) ./ (2*sqrt(c.*cur_p)));
            part2 = sum(sqrt(c.*cur_p));
            
           
            new_g = part1 / part2;  
            
            
            % prevent errors 
            % this may cause the gradient to go in a slightly different direction
            if(isnan(new_g) || isinf(new_g) || ~isreal(new_g)); new_g = 0; end;

            
            % real to prevent rounding errors
            g(k) = g(k) - weights(p) * real(new_g);
        end
    end
          
    
end