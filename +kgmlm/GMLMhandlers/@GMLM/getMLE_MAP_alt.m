%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [paramStruct_opt, resultsStruct_opt, compute_MAP] = getMLE_MAP(obj, paramStruct_init, msgStr, max_iters)

    if(nargin < 2 || isempty(paramStruct_init) || ~isstruct(paramStruct_init))
        compute_MAP = false;
    elseif(isstruct(paramStruct_init) && isfield(paramStruct_init,'Groups') && numel(paramStruct_init.Groups) == obj.dim_J && isfield(paramStruct_init,'H'))
        totalH = numel(paramStruct_init.H);
        totalH_gmlm = obj.dim_H;
        for jj = 1:obj.dim_J
            if(isfield(paramStruct_init.Groups(jj), 'H'))
                totalH     = totalH + numel(paramStruct_init.Groups(jj).H);
            else
                totalH = nan;
                break;
            end
            totalH_gmlm = totalH_gmlm + obj.Groups(jj).dim_H;
        end

        if(totalH ~= totalH_gmlm)
            compute_MAP = false;
        else
            compute_MAP = true;
        end

    else
        compute_MAP = false;
    end

    if(nargin < 3 || isempty(msgStr))
        msgStr = '';
    end
    if(nargin < 4 || isempty(max_iters))
        max_iters = 10e3;
    end
    
    learning_rate.init  = 1e-2;
    learning_rate.decay = 5000;
    learning_rate.min   = 1e-4;
    
    start_time = tic;

    if(nargin > 1 && ~isempty(paramStruct_init) && isstruct(paramStruct_init) && isfield(paramStruct_init,'W') && ~isempty(paramStruct_init.W))
        paramStruct = paramStruct_init;
    else
        paramStruct = obj.getRandomParamStruct([],compute_MAP);
    end

    %optimize all the parameters together
    optsStruct_all  = obj.getEmptyOptsStruct(true, false);
    optsStruct_all = obj.getParamCount(optsStruct_all); 

    %%
    theta_0 = obj.vectorizeParamStruct(paramStruct,optsStruct_all);
    nlpost_theta = @(vv)obj.vectorizedNLL_func(vv, paramStruct, optsStruct_all, compute_MAP);

    [~,~,~,resultsStruct] = nlpost_theta(theta_0);
    fprintf('Starting GMLM optimization: fval init = %e\t\t%s\n',resultsStruct.log_post,msgStr);

    TW_map = quickAdamScript(w_init, lfun, max_iters, learning_rate);
    
    [~,~,paramStruct_opt,resultsStruct_opt] = nlpost_theta(TW_map);
    
    
    
    %% if doing MLE, renormalize all Ts that were just fit
    if(~compute_MAP)
        paramStruct_normalized = obj.normalizeParams(paramStruct_opt);

        rs = obj.computeLL(paramStruct_normalized,optsStruct_empty,true);
        if((rs.log_like_0 - resultsStruct_opt.log_like_0) < -1e-1)
            warning('log likelihood changed after renormalization! (optimization %d)',ss);
        else
            paramStruct_opt = paramStruct_normalized;
        end
    end

    %% print out results
    end_time = toc(start_time);
    fprintf('  maximum iterations reached! Total optimization time = %.1f\n',end_time);

    if(isfield(paramStruct_opt, 'W_all'))
        paramStruct_opt = rmfield(paramStruct_opt, 'W_all');
    end
    if(isfield(paramStruct_opt, 'H_all'))
        paramStruct_opt = rmfield(paramStruct_opt, 'H_all');
    end
end

%% adam for GD
% using Adam here for gradient descent - experimentally works way better than Quasi-Newton methods so far
%   would also extend to stochastic gradients
function [w_best, L, learning_rate, w] = quickAdamScript(w_init, lfun, N, learning_rate)
% learning_rate = exp(-(0:(N-1))./learning_rate.decay) * (learning_rate.init - learning_rate.min) + learning_rate.min;
learning_rate = linspace(learning_rate.init, learning_rate.min, N);

beta_1=0.9;
beta_2=0.999;
epsilon=1e-07;

P = numel(w_init);

w = w_init;

L = nan(N,1);

m = zeros(P,1);
v = zeros(P,1);
w_best = w;

for tt = 1:N
    [L(tt), g] = lfun(w);
    if(mod(tt, 1e3) == 0)
        fprintf('\titer %d / %d\n', tt, N);
    end
    
    m = beta_1*m + (1-beta_1)*g;
    v = beta_2*v + (1-beta_2)*g.^2;

    m_hat = m./(1-beta_1.^tt);
    v_hat = v./(1-beta_2.^tt);

    w  = w - learning_rate(tt)*m_hat./(sqrt(v_hat) + epsilon);
    
    if(L(tt) == nanmin(L))
        w_best = w;
    end
end

end