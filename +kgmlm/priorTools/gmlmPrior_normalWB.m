% simple i.i.d. standard normal prior (NO hyperparams) over the elements of W and B
function [results] = gmlmPrior_normalWB(results, params)
results.log_prior_WB = -1/2*sum(params.W.^2, 'all') + -1/2*sum(params.B.^2, 'all') ;

if(~isempty(results.dW)) %if is empty, not required to compute derivative, otherwise ADD derivative of log prior to struct
    results.dW = results.dW - params.W;
end
if(~isempty(results.dB))
    results.dB = results.dB - params.B;
end

