%  Package GMLM_dmc for dimensionality reduction of neural data.
%   
%  References
%   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
%   decisions in parietal cortex reflects long-term training history.
%   bioRxiv
%
%  Copyright (c) 2021 Kenneth Latimer
%
%   This software is distributed under the GNU General Public
%   License (version 3 or later); please refer to the file
%   License.txt, included with the software, for details.
%
function [timeBeforeSample, timeAfterTestOrLever, delta_t] = getDefaultTrialSettings(TaskInfo, printWarning1, printWarning2, printWarning3)

timeBeforeSample     = 0;
if(nargin < 1 || isempty(TaskInfo))
    delta_t = 5e-3;
else
    delta_t = TaskInfo.binSize_ms * 1e-3;
end

timeAfterTestOrLever = ceil(50e-3 / delta_t);

if(nargin > 1 && printWarning1)
    warning('Using default timing settings: timeBeforeSample = 0 ms');
end
if(nargin > 2 && printWarning2)
    warning('Using default timing settings: timeAfterTestOrLever = 50 ms');
end
if(nargin > 3 && printWarning3 && (nargin < 1 || ~isstruct(TaskInfo) || ~isfield(TaskInfo, 'binSize_ms') || isempty(Taskinfo.binSize_ms)))
    warning('Using default timing settings: delta_t = 5 ms');
end