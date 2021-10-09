function X = gmpd_simplex_proj(Y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [N,D] = size(Y);
    X = sort(Y,2,'descend');
    Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
    X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
end

