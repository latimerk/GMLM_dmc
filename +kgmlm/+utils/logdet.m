function [logdetA, G_chol] = logdet(G)
if(~isdiag(G))
    try 
        G_chol = chol(G);
        logdetA = 2*sum(log(diag(G_chol)));
    catch
        logdetA = log(max(1e-20,det(G)));
    end
else
    logdetA = sum(log(diag(G)));
    G_chol  = diag(sqrt(diag(G)));
end