function sod = gmpd_uniform_sod(p,set_p,weights)
%GMPD_UNIFORM_SOD computes the SOD of a given alpha for the Kolmogorov or Uniform metric
    
    % TODO look for correct summation?
    sod = sum(weights .* max(abs( repmat(p,[size(set_p,1),1]) - set_p),[],2),1);
end