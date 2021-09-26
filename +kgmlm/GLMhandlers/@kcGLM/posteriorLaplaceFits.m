function [M_chol] = posteriorLaplaceFits(obj, paramStruct, M_chol, s_idx, computeChol)

w_opt = obj.getMAP(paramStruct);
[~,~,M_est] = obj.computeNLPost_params(w_opt,paramStruct,false);

if(computeChol)
    mc = chol(M_est);
else
    mc = M_est;
end
M_chol.valid(s_idx) = all(~isinf(mc(:)) & ~isnan(mc(:)));
M_chol.W(:,:,s_idx) = mc;
end
