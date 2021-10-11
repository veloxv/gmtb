function sod = gmpd_jensen_shannon_sod(p,set_p,weights)
%GMPD_JENSEN_SHANNON_SOD computes the SOD of a given alpha for the Jensen-Shannon divergence
    
    sod = 0;
    
    for p_i = 1:size(set_p,1)
        pM = (p + set_p(p_i,:))/2;
        sod = sod + weights(p_i) * (gmpd_kullback_leibler1_dist(p,pM) + gmpd_kullback_leibler1_dist(set_p(p_i,:),pM))/2;
    end
    
end