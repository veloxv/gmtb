function sod = gmpd_total_variation_sod(p,set_p,weights)
%GMPD_TOTAL_VARIATION_SOD computes the SOD of a given alpha for the Total Variation Distance. 
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        sod = sod + weights(p_i) * 1/2 * sum(abs(p-set_p(p_i,:)));
    end
    
end