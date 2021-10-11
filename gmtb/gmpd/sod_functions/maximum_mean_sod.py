function f = gmpd_maximum_mean_sod(p,set_p,weights)
%GMPD_MAXIMUM_MEAN_SOD computes the SOD of a given alpha for the Maximum Mean Discrepancy
    
    f = 0;
    
    for p_i = 1:size(set_p,1)
        f = f + weights(p_i) * (sum((1:length(p)) .* p) - sum((1:length(p)) .* (set_p(p_i,:)))).^2;
    end
    
end