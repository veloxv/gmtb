function sod = gmpd_chi_squared1_sod(p,set_p,weights)
%GMPD_CHI_SQUARED1_SOD computes the SOD of a given alpha for the Chi^2 distance
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        idx = set_p(p_i,:) > eps; %~= 0;
        sod = sod + weights(p_i) * sum((p(idx) - set_p(p_i,idx)).^2 ./ set_p(p_i,idx));
    end
    
end