%quick and dirty tool for plotting out HMC sample progress
% I didn't document this well, but it's not super imporant
% This prints out stuff I want for the DMC task GMLM, but not every GMLM
function [] = plotSamples(obj, samples, sample)
clf;
TotalSamples = size(samples.H, 2);

if(nargin < 3 || isempty(sample))
    sample = TotalSamples;
end

NR = 2;
NC = max(4, obj.dim_J);

subplot(NR,NC,1)
if(sample > 200)
    plot(101:TotalSamples,samples.log_post(101:end))
else
    semilogy(1:TotalSamples,samples.log_post)
end
xlabel('sample');
ylabel('log posterior');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,2)
if(sample > 200)
    plot(101:TotalSamples,samples.log_like(101:end))
else
    semilogy(1:TotalSamples,samples.log_like)
end
xlabel('sample');
ylabel('log likelihood');
set(gca,'tickdir','out','box','off');

subplot(NR,NC,3)
semilogy(1:TotalSamples,samples.e)
xlabel('sample');
ylabel('HMC step size');
set(gca,'tickdir','out','box','off');


subplot(NR,NC,4)
hold on
if(~isempty(samples.H))
    plot(1:TotalSamples,samples.H)
end
hold off
set(gca,'tickdir','out','box','off');
title('H');

for jj = 1:obj.dim_J
    subplot(NR,NC,NC + jj)
    plot(1:TotalSamples,samples.Ks{jj})
    xlabel('sample');
    ylabel('coefficients');
    set(gca,'tickdir','out','box','off');
    title(obj.GLMstructure.group_names(jj))
end

