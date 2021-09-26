%quick and dirty tool for plotting out HMC sample progress
% I didn't document this well, but it's not super imporant
% This prints out stuff I want for the DMC task GMLM, but not every GMLM
function [] = plotSamples(obj, samples, paramStruct, sample)
clf;
TotalSamples = size(samples.W, 2);

if(nargin < 4 || isempty(sample))
    sample = TotalSamples;
end

sampleStart = 1;
if(sample >= 2000)
    sampleStart = 1000;
elseif(sample >= 1000)
    sampleStart = 500;
elseif(sample >= 500)
    sampleStart = 300;
elseif(sample >= 300)
    sampleStart = 200;
elseif(sample >= 200)
    sampleStart = 100;
elseif(sample >= 100)
    sampleStart = 20;
end

NR = 1+length(samples.Groups);
NC = 5;
subplot(NR,NC,1)
plot(1:TotalSamples,samples.W)
xlabel('sample');
ylabel('baseline rates (W)');
set(gca,'tickdir','out','box','off');

if(size(samples.B,1) > 0)
    subplot(NR,NC,2)
    plot(1:TotalSamples,squeeze(samples.B(:,1,:)))
    xlabel('sample');
    ylabel('full-rank terms (B)');
    set(gca,'tickdir','out','box','off');
end

subplot(NR,NC,3)
if(sample > 200)
    plot(    sampleStart:TotalSamples, samples.log_post(sampleStart:end))
else
    semilogy(sampleStart:TotalSamples, samples.log_post(sampleStart:end))
end
xlabel('sample');
ylabel('log posterior');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,4)
if(sample > 200)
    plot(    sampleStart:TotalSamples, samples.log_like(sampleStart:end))
else
    semilogy(sampleStart:TotalSamples, samples.log_like(sampleStart:end))
end
xlabel('sample');
ylabel('log likelihood');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,5)
semilogy(1:TotalSamples,samples.e)
xlabel('sample');
ylabel('HMC step size');
set(gca,'tickdir','out','box','off');

for jj = 1:length(samples.Groups)
    subplot(NR,NC,1+jj*NC)
    plot(1:TotalSamples,squeeze(samples.Groups(jj).V(:,1,:)))
    title(sprintf('%s, rank = %d',paramStruct.Groups(jj).name,  size(paramStruct.Groups(jj).T{1},2)));
    set(gca,'tickdir','out','box','off');
    
    for ss = 1:min(3, numel(samples.Groups(jj).T))
        subplot(NR,NC,1 + ss +jj*NC)
        plot(1:TotalSamples,squeeze(samples.Groups(jj).T{ss}(:,1,:)));
        set(gca,'tickdir','out','box','off');
        title(sprintf('dim = %s',paramStruct.Groups(jj).dim_names(ss)));
    end    
        
    if(numel(samples.Groups(jj).T) == 1 && size(paramStruct.Groups(jj).T{1},2) > 1)
        
        subplot(NR,NC,3+jj*NC)
        plot(1:TotalSamples,squeeze(samples.Groups(jj).V(:,2,:)))
        title('second component');
        set(gca,'tickdir','out','box','off');
        
        subplot(NR,NC,4+jj*NC)
        plot(1:TotalSamples,squeeze(samples.Groups(jj).T{1}(:,2,:)))
        set(gca,'tickdir','out','box','off');
        title(sprintf('dim = %s, second component', paramStruct.Groups(jj).dim_names(1)));
    end
end


subplot(NR,NC,NC + NC)
hold on
if(~isempty(samples.H))
    plot(1:TotalSamples,samples.H)
end
for jj = 1:length(samples.Groups)
    if(~isempty(samples.Groups(jj).H))
        plot(1:TotalSamples,samples.Groups(jj).H)
    end
end
hold off
set(gca,'tickdir','out','box','off');
title('H');

