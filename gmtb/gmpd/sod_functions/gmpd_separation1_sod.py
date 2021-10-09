function sod = gmpd_separation1_sod(p,set_p,weights)
%GMPD_SEPARATION1_SOD computes the SOD of a given alpha for the Separation distance
    
    
    % TODO look for correct summation?
    sod = sum(weights .* max(1-( repmat(p,[size(set_p,1),1]) ./ set_p),[],2),1);
end