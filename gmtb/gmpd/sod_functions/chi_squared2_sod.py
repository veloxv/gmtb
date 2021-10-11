function sod = gmpd_chi_squared2_sod(p,set_p,weights)
%GMPD_CHI_SQUARED2_SOD computes the SOD of a given alpha for the Chi^2 distance
    
    sod = 0;
    
    idx = p > eps; % ~= 0 does not always work!
     
    for p_i = 1:size(set_p,1)      
        sod = sod + weights(p_i) * sum((set_p(p_i,idx) - p(idx)).^2 ./ p(idx));
    end
    
end