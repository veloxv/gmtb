function g = gmpd_separation1_grad(alpha,set_p,weights)
%GMPD_SEPARATION_GRAD computes the gradient of the Separation SOD at point alpha

    g = zeros(1,length(alpha));
    n = length(alpha)+1;    
    c = [alpha,0] + (1 - sum(alpha))/n;
    
    
    
    for k = 1:length(alpha)
        for p = 1:size(set_p,1)
            
            cur_p = set_p(p,:);
            
            % max index
            %[~, m] = max(1-(c./cur_p));
            
            g(k) = g(k) + weights(p) * sum(- (-1/n  + (k==m)) ./ cur_p(k));
           
        end
    end
          
    
    
end