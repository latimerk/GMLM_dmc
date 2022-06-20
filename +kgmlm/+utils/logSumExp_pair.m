function [log_m] = logSumExp_pair(log_x, log_y)

cs = max(log_x, log_y);
log_x = log_x - cs;
log_y = log_y - cs;

log_m = cs + log(exp(log_x) + exp(log_y));