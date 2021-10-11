function sod = gmpd_bhattacharyya_sod(p,set_p,weights)
%GMPD_BHATTACHARYYA_SOD computes the SOD of a given alpha for the Bhattacharyya distance
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        sod = sod + weights(p_i) * -log(real(sum(sqrt(p.*set_p(p_i,:)))));
    end
    
end