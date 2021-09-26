%Function used to compute the log evidence (sort of --- I also have a prior over the hyperparams, so it's an approximate log posterior over the hyperparams)
% where the evidence is computed using a Laplace approximation of the posterior over the parameters and plugging into Bayes' rule
function [nle, ndle, params, results, ld, d_ld, sigma] = computeNLEvidence(obj, H, params, trial_weights)
    params.H(:) = H;
    if(nargin < 4)
        trial_weights = [];
    end
    
    params = obj.computeMAP(params, "trial_weights", trial_weights, "display", 'none');    
    
    if(nargout > 1)
        opts    = obj.getComputeOptionsStruct(true, "trial_weights", trial_weights, "includeHyperparameters", true);
        results = obj.computeLogPosterior(params, opts, true);
        nlp   = -results.log_post;
        sigma = -results.d2K;
        dH_npost = -results.dH;
        
        d_ld = zeros(obj.dim_H,1);
        if(all(~isinf(sigma), 'all') && all(~isnan(sigma),'all'))
            for ii = 1:obj.dim_H
                rc = rcond(sigma);
                % derivative of the log posterior over the parameters w.r.t. each hyperparam (log posterior is laplace approximation evaluated at MAP: this is
               % derivative of the log determinant term in the Gaussian PDF)
                if(~isnan(rc) && ~isinf(rc) && rc > 1e-16)
                    d_ld(ii) = 1/2*trace(sigma\results.dprior_sigma_inv(:,:,ii));
                else
                    d_ld(ii) = 1/2*trace(pinv(sigma)*results.dprior_sigma_inv(:,:,ii));
                end
                
                %does a lot of checks for invalid values: makes sure the prior isn't spitting out total nonsense
                if(isinf(d_ld(ii)) || isnan(d_ld(ii)))
                    if(isinf(d_ld(ii)))
                        fprintf('d_ld(%d) is inf\n', ii);
                    else
                        fprintf('d_ld(%d) is nan\n', ii);
                    end
                    fprintf(' sigma inf %d, sigma nan %d, dprior nan %d, dprior inf %d, rc = %f\n', sum(isinf(sigma(:))), sum(isnan(sigma(:))), sum(sum(isnan(results.dprior_sigma_inv(:,:,ii)))),sum(sum(isinf(results.dprior_sigma_inv(:,:,ii)))), rcond(sigma));
                end
            end
        else
            fprintf('sigma for evidence optimization invalid (contains inf/nan)\n');
            d_ld(:) = nan;
        end
        ndle = dH_npost + d_ld;

        bc = ~isnan(ndle) | ~isinf(ndle);
        if(~all(bc))
            fprintf('dH_post contains inf/nan\n');
            bc = find(~bc);
            for ii = 1:numel(bc)
                fprintf('bad entries: %d, H = %.4f\n', ii, params.H(ii));
            end
            %values to steer the optimizer away without crashing
            ndle(:) = 0;
            nlp = inf;
        end
    else
        opts    = obj.getComputeOptionsStruct(true, "trial_weights", trial_weights, "includeHyperparameters", false);
        results = obj.computeLogPosterior(params, opts);
        nlp     = -results.log_post;
        sigma   = -results.d2K;
    end

    if(all(~isinf(sigma), 'all') && all(~isnan(sigma),'all'))
        ld = 1/2*kgmlm.utils.logdet(sigma);
        nle = nlp + ld;
    else
        fprintf('sigma for evidence optimization invalid (contains inf/nan)\n');
        nle = inf;
    end

    if(~all(~isnan(nle)) || ~all(~isinf(nle)))
        fprintf('nle showing inf/nan\n');
    end
end