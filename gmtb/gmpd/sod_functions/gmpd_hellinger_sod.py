function sod = gmpd_hellinger_sod(p,set_p,weights)
%GMPD_HELLINGER_SOD computes the SOD of a given alpha for the Hellinger distance
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        sod = sod + weights(p_i) * sqrt(1-sum(sqrt(p .* set_p(p_i,:))));
    end
    
end