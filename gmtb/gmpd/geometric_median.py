function [ med ] = gmpd_geometric_median( set_p, threshold, max_iter )
%GMPD_GEOMETRIC_MEDIAN caluclates the geometric median between points in euclidean vector space
% using the Weiszfeld algorithm.
% threshold = termination threshold of the weiszfeld algorithm

if (nargin < 3)
    max_iter = 10000;
end

if (nargin < 2)
    threshold = 1e-10;
end

if (isnan(set_p))
    med = NaN;
    return;
end


%% Weizfeld algorithm        

[m,n] = size(set_p);
y_old = zeros(1,n);
y_new = sum(set_p)/n;
iter = 0;

while (sqrt(sum((y_new - y_old).^2)) > threshold && iter < max_iter)
    iter = iter + 1;
    y_old = y_new;
    eukl_dist = sqrt(sum((set_p - repmat(y_old,m,1)).^2,2));
    numerator = sum(set_p ./ repmat(eukl_dist,1,n),1);
    denominator = sum(1 ./ eukl_dist);
    y_new = numerator ./ denominator;
end
med = y_new;
 


end


