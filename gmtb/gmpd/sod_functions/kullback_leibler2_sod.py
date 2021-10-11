function sod = gmpd_kullback_leibler2_sod(p,set_p,weights)
%GMPD_KULLBACK_LEIBLER2_SOD computes the SOD of a given alpha for the Kullback Leibler divergence. 
% This Version assumes the median is p2 in the function
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        sod = sod + weights(p_i) * sum(real(set_p(p_i,:) .* log(set_p(p_i,:)./p)));
    end
    
end