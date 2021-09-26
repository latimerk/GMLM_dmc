function [] = plotSamples_glm(samples,HMC_settings, paramStruct, sample)
clf;
if(nargin < 4 || isempty(sample))
    sample = HMC_settings.nTotal;
end

NR = 2;
NC = 3;

subplot(NR,NC,1)
if(sample > 300)
    plot(201:HMC_settings.nTotal,samples.log_post(201:end))
else
    semilogy(1:HMC_settings.nTotal,samples.log_post)
end
xlabel('sample');
ylabel('log posterior');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,2)
if(sample > 300)
    plot(201:HMC_settings.nTotal,samples.log_like(201:end))
else
    semilogy(1:HMC_settings.nTotal,samples.log_like)
end
xlabel('sample');
ylabel('log likelihood');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,3)
semilogy(1:HMC_settings.nTotal,squeeze(samples.e))
xlabel('sample');
ylabel('HMC step size');
set(gca,'tickdir','out','box','off');        

subplot(NR,NC,1+NC)
plot(1:HMC_settings.nTotal,samples.W)
xlabel('sample');
ylabel('W');
set(gca,'tickdir','out','box','off');

if(~isempty(samples.H))
    subplot(NR,NC,NC + 2)
    hold on
    plot(1:HMC_settings.nTotal,samples.H)
    
    hold off
    set(gca,'tickdir','out','box','off');
    title('H')
end
