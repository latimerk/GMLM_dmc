%% A quick and dirty diagnotistic script to check log likelihood derivatives to make sure the optimized CUDA code is actually working. Runs through a bunch of setup scenarios.
% I've occasionally noticed failures that are due to a random dataset/parameters being numerically unstable (exploding values). This doesn't mean the derivative
% is incorrect.
function [params] = runDerivativeChecks(testType, isPop, deviceNumbers, dim_A, shrink_dim_R, pauseAtParts, params, use_posterior)

if(nargin < 2 || isempty(isPop))
    isPop = false;
end
if(nargin < 3 || isempty(deviceNumbers))
    deviceNumbers = [0 0];
end
if(nargin < 4 || isempty(dim_A))
    dim_A = 2;
end
if(nargin < 5 || isempty(shrink_dim_R))
    shrink_dim_R = false;
end
if(nargin < 6 || isempty(pauseAtParts))
    pauseAtParts = false;
end

if(isPop)
    generator = @kgmlm.diagnostics.constructGMLMPopforTests;
else
    generator = @kgmlm.diagnostics.constructGMLMforTests;
end

if(nargin < 8 || isempty(use_posterior))
    use_posterior = true;
end

if(isnumeric(testType))
    switch(testType)
        case 1
            % group with 1 dim (local)
            [GMLMstructure, trials] = generator( 1, false, false, dim_A);

        case 2
            % group with 1 dim (shared)
            [GMLMstructure, trials] = generator( 1, true, false, dim_A);

        case 3
            % group with 2 dims - independent factors (one local, one shared)
            [GMLMstructure, trials] = generator( [1 2], [false true], [false false], dim_A);

        case 4
            % group with 2 dims - independent factors (one local, one shared identity)
            [GMLMstructure, trials] = generator( [1 2], [false true], [false true], dim_A);

        case 5
            % group with 4 dims - factors 1:2, 3:4 (local, local)
            [GMLMstructure, trials] = generator( [1 1 2 2], [false true], [false false], dim_A);

        case 6
            % group with 4 dims - factors 1:2, 3:4 (local, shared identity)
            [GMLMstructure, trials] = generator( [1 1 2 2], [false true], [false true], dim_A);

        case 7
            % group with 4 dims - factors 1:3, 4, (local, shared)
            [GMLMstructure, trials] = generator( [1 1 1 2], [false true], [false false], dim_A);

        case 8
            % group with 4 dims - factors 1:2, 3:4 (local, shared identity) - but local coefficients are the same for all events
            [GMLMstructure, trials] = generator( [1 1 2 2], [false true], [false false], dim_A);
            for tt = 1:numel(trials)
                trials(tt).Groups(1).X_local{1} = trials(tt).Groups(1).X_local{1}(:, :, 1);
            end
        case 9
            % group with 2 dims - independent factors (one local, one shared identity)
            [GMLMstructure, trials] = generator( [1 2], [false true], [false false], dim_A);
            for tt = 1:numel(trials)
                trials(tt).Groups(1).X_local{1} = trials(tt).Groups(1).X_local{1}(:, :, 1);
            end
        case 10
            % group with 2 dims - independent factors (two local)
            [GMLMstructure, trials] = generator( [1 2], [false false], [false false], dim_A);
            for tt = 1:numel(trials)
                trials(tt).Groups(1).X_local{1} = trials(tt).Groups(1).X_local{1}(:, :, 1);
            end
        case 11
            % group with 4 dims - independent factors (two local, one shared, one shared identity)
            [GMLMstructure, trials] = generator( [1 2 3 4], [false false true true], [false false false true], dim_A);
            for tt = 1:numel(trials)
                trials(tt).Groups(1).X_local{1} = trials(tt).Groups(1).X_local{1}(:, :, 1);
            end
        otherwise
            error("Test not found");
    end


    gmlm = kgmlm.GMLM(GMLMstructure, trials, 1);
    pauseMessage(pauseAtParts, "Finished contructing GMLM...\n"); 
else
    gmlm = testType;
    fprintf("Using provided GMLM.\n");
end
if(shrink_dim_R)
    fprintf("Group dimension is %d.\n", gmlm.dim_R(1));
    gmlm.setDimR(1, gmlm.dim_R(1) - 1);
    fprintf("Group dimension set to %d.\n", gmlm.dim_R(1));
end

if(~gmlm.isOnGPU())
    gmlm.toGPU(deviceNumbers);
    pauseMessage(pauseAtParts, "Finished sending GMLM to GPU...\n");
else
    fprintf("GMLM is already on device.\n");
end

%% test comps
if(nargin < 7 || isempty(params))
    params     = gmlm.getRandomParamStruct();
end
% params.Groups(1).T{1} = params.Groups(1).T{1}./10;
% params.Groups(1).T{2} = params.Groups(1).T{2}./10;
% params.Groups(1).T{3} = params.Groups(1).T{3}./10;
% params.Groups(1).T{4} = params.Groups(1).T{4}./10;
opts = gmlm.getComputeOptionsStruct(false);


[results_est, results_all, ll_host, params] = kgmlm.diagnostics.checkDerivatives(gmlm, params, [], [], use_posterior);

% no derivatives
results = gmlm.computeLogPosterior(params, opts); %#ok<*NASGU>
pauseMessage(pauseAtParts, "Computed LL...\n");

% add dW
opts.dW = true;
results = gmlm.computeLogPosterior(params, opts);
pauseMessage(pauseAtParts, "Computed dLL/dW...\n"); 

% add dB
opts.dB = true;
results = gmlm.computeLogPosterior(params, opts);
pauseMessage(pauseAtParts, "Computed dLL/dB...\n"); 

% for each group
for jj = 1:numel(opts.Groups)
    % test dV
    opts.Groups(jj).dV = true;
    results = gmlm.computeLogPosterior(params, opts);
    pauseMessage(pauseAtParts, sprintf("Computed dLL/dV[%d]...\n", jj));
    % add each dT
    for ss = 1:numel(params.Groups(jj).T)
        opts.Groups(jj).dT(ss) = true;
        results = gmlm.computeLogPosterior(params, opts);
        pauseMessage(pauseAtParts, sprintf("Computed dLL/dT[%d,%d]...\n", jj,ss));
    end
end
results = gmlm.computeLogPosterior(params, opts);
    
%% run derivative checker
figure(1);
clf;
plotDerivativeComparison(results_all, results_est, ll_host, use_posterior);
pauseMessage(pauseAtParts, sprintf("Done with main LL...\n"));
%return;

%% run derivative check with weights

%random weights - full full
[results_est, results_all, ll_host] = kgmlm.diagnostics.checkDerivatives(gmlm, params, gmlm.dim_M, [], use_posterior);

figure(2);
clf;
plotDerivativeComparison(results_all, results_est, ll_host, use_posterior);
pauseMessage(pauseAtParts, sprintf("Done with weighted LL 1...\n"));

% some subset of weights
[results_est, results_all, ll_host] = kgmlm.diagnostics.checkDerivatives(gmlm, params, 10, [], use_posterior);

figure(3);
clf;
plotDerivativeComparison(results_all, results_est, ll_host, use_posterior);
pauseMessage(pauseAtParts, sprintf("Done with weighted LL 2...\n"));

if(gmlm.isSimultaneousPopulation   )
    %random weights - full full
    [results_est, results_all, ll_host] = kgmlm.diagnostics.checkDerivatives(gmlm, params, gmlm.dim_M, true, use_posterior);
    figure(4);
    clf;
    plotDerivativeComparison(results_all, results_est, ll_host, use_posterior);
    pauseMessage(pauseAtParts, sprintf("Done with weighted LL 1...\n"));

    % some subset of weights
    [results_est, results_all, ll_host] = kgmlm.diagnostics.checkDerivatives(gmlm, params, 10, true, use_posterior);

    figure(5);
    clf;
    plotDerivativeComparison(results_all, results_est, ll_host, use_posterior);
    pauseMessage(pauseAtParts, sprintf("Done with weighted LL 2...\n"));
end

end

%%
function [] = pauseMessage(pauseAtParts, msg)
    if(pauseAtParts)
        fprintf(msg);
        pause();
    end
end

%% plot a comparison
function [] = plotDerivativeComparison(results_all, results_est, ll_host, use_posterior)
    
    NR = numel(results_all.Groups) + 1;
    NC = max(5, max(arrayfun(@(aa) numel(aa.dT), results_all.Groups)) + 1 + use_posterior);
    
    subplot(NR, NC, 1);
    plot([ ll_host(:) results_all.trialLL(:)])
    legend(["Host", "GPU"]);
    title('trial LLs');

    subplot(NR, NC, 2);
    dds = ll_host(:) - results_all.trialLL(:);
    plot(dds)
    title('trial LL diffs');
    if(max(abs(dds)) < 10e-8)
        ylim([-1 1]*10e-8);
    end


    subplot(NR, NC, 3);
    plot([results_all.dW(:) results_est.dW(:)])
    legend(["GPU", "est"]);
    title('dW');

    subplot(NR, NC, 4);
    plot([results_all.dB(:) results_est.dB(:)])
    title('dB');

    if(use_posterior && isfield(results_all, "dH"))
        subplot(NR, NC, 5);
        plot([results_all.dH(:) results_est.dH(:)])
        title('dH');
    end

    for jj = 1:numel(results_all.Groups)
        subplot(NR, NC, 1 + jj*NC);
        cla
        dV = results_all.Groups(jj).dV';
        dV2 = results_est.Groups(jj).dV';
        plot([dV(:) dV2(:)])
        title(sprintf('Groups(%d).dV', jj));

        S = numel(results_est.Groups(jj).dT);
        for ss = 1:S
            subplot(NR, NC, 1 + ss + jj*NC);
            cla;
            plot([results_all.Groups(jj).dT{ss}(:) results_est.Groups(jj).dT{ss}(:)])
            title(sprintf('Groups(%d).dT{%d}', jj, ss));
        end

        if(use_posterior && isfield(results_all.Groups(jj), "dH"))
            subplot(NR, NC, 1 + S + 1 + jj*NC);
            plot([results_all.Groups(jj).dH(:) results_est.Groups(jj).dH(:)])
            title('dH');
        end
    end
end
