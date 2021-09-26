%% setup test data

rng(20200519); % set seed

N = 2000;%(1500/5)*500*5; %total number of observations
M = 7;   %number of neurons

dt = 0.7; %making this not equal to 1 as a test

% 3-Tensors


R1 = 4;
R2 = 3;
R3 = 2;
R4 = 3;

% generate F1 (ff_1 x pp_1)
ff_1 = 25;
pp_1 = 12;

F1 = randn(ff_1,pp_1);
F1 = F1./sqrt(sum(F1.^2));

% generate A1 (N x 1)
A1 = mod(0:(N-1),ff_1)'+1;

% generate G1 (gg_1 x qq_1)
gg_1 = 20;
qq_1 = 10;

G1 = randn(gg_1,qq_1);

% generate B1 (N x 1)
B1 = randi(gg_1,[N 1]);

% generate F2 (ff_2 x pp_2)
ff_2 = 30;
pp_2 = 5;

F2 = randn(ff_2,pp_2);

% generate A2 (N x 2)
A2 = randi(ff_2,[N 2]);

% generate G2 (gg_2 x qq_2)
gg_2 = 15;
qq_2 = 4;

G2 = randn(gg_2,qq_2);

% generate B2 (N x 2)
B2 = randi(gg_2+5,[N 2])-5; %make some indices negative: test for invalid indices (should work as assumed 0 entries)


% 2-Tensors

% generate F3 (ff_3 x pp_3)
ff_3 = 50;
pp_3 = 14;
F3 = randn(ff_3,pp_3);

% generate A3 (N x 1)
A3 = randi(ff_3,[N 1]);


% generate F4 (N * 2 x P4) (note: A4 is empty)
P4 = 8;
F4 = orth(randn(N*2,P4));


% generate M_idx (start times for each neuron)
M_idx = round(linspace(1,N+1,M+1));
M_idx = M_idx(:);

% generate T_idx
T_idx = sort([M_idx; round(diff(M_idx)/2+M_idx(1:M))]);

M_idx = M_idx(1:end-1);
T_idx = T_idx(1:end-1);

%% setup test covariates
K = 30;

% generate W (M x 1)
W = randn(M,1);

% 3-Tensors

% generate T1 (pp_1 x R1)
T1 = randn(pp_1,R1);

% generate U1 (qq_1 x R1)
U1 = randn(qq_1,R1);

% generate V1 (M  x R1)
V1 = randn(M ,R1)./K;

% generate T2 (pp_2 x R2)
T2 = randn(pp_2,R2);

% generate U2 (qq_2 x R2)
U2 = randn(qq_2,R2);

% generate V2 (M  x R2)
V2 = randn(M ,R2)./K;


% 2-Tensors

% generate T3 (pp_3 x R3)
T3 = randn(pp_3,R3);

% generate V3 (M  x R3)
V3 = randn(M ,R3)./K;

% generate T4 (P4 x R4)
T4 = randn(P4,R4);

% generate V4 (M  x R4)
V4 = randn(M ,R4)./K;


%% sim with tensor 1

% generate samples

% select init point

% get LL: on CPU

% get GPU LL & derivatives

% test derivatives with finite difference



%% sim with tensor 2

% generate samples

% select init point

% get LL: on CPU

% get GPU LL & derivatives

% test derivatives with finite difference



%% sim with tensor 3

% generate samples

% select init point

% get LL: on CPU

% get GPU LL & derivatives

% test derivatives with finite difference


%% sim with tensor 4


% generate samples

% select init point

% get LL: on CPU

% get GPU LL & derivatives

% test derivatives with finite difference



%% sim with tensors 1, 2, 3, 4
mm1 = 1;
mm2 = 1;
mm3 = 1;
mm4 = 1;
mmw= 1;
Ts = {T1*mm1,T2*mm2,T3*mm3,T4*mm4};
Us = {U1*mm1,U2*mm2,[],[]};
Vs = {V1*mm1,V2*mm2,V3*mm3,V4*mm4};

As = {A1,A2,A3,[]};
Bs = {B1,B2,[],[]};

Fs = {F1,F2,F3,F4};
Gs = {G1,G2,[],[]};
    
% generate samples
[lambda_0,Y,ll_0,~,ll_indiv_0] = gmlmRatesForTest(W,Ts,Us,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, [],dt);


%%
idx = 1e5;
dx = 1./idx;

coeffs    = [-1/60 3/20 -3/4 0 3/4 -3/20 1/60];
dxs       = [-3   -2    -1   0 1    2    3]*dx;

coeffs2     = zeros(5,5);
coeffs2(4,1) = 8;
coeffs2(5,2) = 8;
coeffs2(1,4) = 8;
coeffs2(2,5) = 8;

coeffs2(2,1) = -8;
coeffs2(1,2) = -8;
coeffs2(4,5) = -8;
coeffs2(5,4) = -8;

coeffs2(5,1) = -1;
coeffs2(1,5) = -1;
coeffs2(1,1) =  1;
coeffs2(5,5) =  1;

coeffs2(2,2) =  64;
coeffs2(4,4) =  64;
coeffs2(2,4) = -64;
coeffs2(4,2) = -64;

coeffs2 = coeffs2*(idx./12)^2;

dxs2 = (-2:2)*dx;

if(N > 5e3)
    return;
end
%%


der_ests.W  = nan(size(W));
der_ests.W2 = nan(numel(W),numel(W));
ff = @(ww) gmlmRatesForTest(ww,Ts,Us,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);

for ii = 1:numel(W)
    lls    = nan(numel(coeffs),1);
    for jj = 1:numel(coeffs)
        W_c = W;
        W_c(ii) = W_c(ii) + dxs(jj);
        [~,~,lls(jj)] = gmlmRatesForTest(W_c,Ts,Us,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
    end
    der_ests.W(ii) = (coeffs*lls)./dx;
    
%     for jj = ii:numel(W)
%         x_0 = W;
%         fs = zeros(length(dxs2),length(dxs2));
%         for kk = 1:length(dxs2)
%             for ll = 1:length(dxs2)
%                 if(coeffs2(kk,ll) ~= 0)
%                     xx_c = x_0;
%                     xx_c(ii) = xx_c(ii) + dxs2(kk);
%                     xx_c(jj) = xx_c(jj) + dxs2(ll);
%                     [~,~,fs(kk,ll)] = ff(xx_c);
%                 else
%                     fs(kk,ll) = 0;
%                 end
%             end
%         end
%         der_ests.W2(ii,jj) = sum(sum(fs.*coeffs2));
%         der_ests.W2(jj,ii) = der_ests.W2(ii,jj);
%     end
end
der_ests.W2(abs(der_ests.W2)<1e-5) = 0;


    
der_ests.V = cell(4,1);
for dd = 1:4
    der_ests.V{dd} = nan(size(Vs{dd}));
    der_ests.V2{dd} = nan(numel(Vs{dd}),numel(Vs{dd}));
    % coeffs = [-1 1];
    % dxs    = [0 1 ]*dx;
    
    for ii = 1:numel(Vs{dd})
        lls    = nan(numel(coeffs),1);
        for jj = 1:numel(coeffs)
            V1_c = Vs{dd};
            V1_c(ii) = V1_c(ii) + dxs(jj);
            Vs_c = Vs;
            Vs_c{dd} = V1_c;
            [~,~,lls(jj)] = gmlmRatesForTest(W,Ts,Us,Vs_c,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
        end
        der_ests.V{dd}(ii) = (coeffs*lls)./dx;
    
%         for jj = ii:numel(Vs{dd})
%             x_0 = Vs{dd};
%             fs = zeros(length(dxs2),length(dxs2));
% 
%             for kk = 1:length(dxs2)
%                 for ll = 1:length(dxs2)
%                     if(coeffs2(kk,ll) ~= 0)
%                         Vs_c = Vs;
%                         Vs_c{dd}(ii) = Vs_c{dd}(ii) + dxs2(kk);
%                         Vs_c{dd}(jj) = Vs_c{dd}(jj) + dxs2(ll);
%                         [~,~,fs(kk,ll)] = gmlmRatesForTest(W,Ts,Us,Vs_c,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
%                     else
%                         fs(kk,ll) = 0;
%                     end
%                 end
%             end
% 
%             der_ests.V2{dd}(ii,jj) = sum(sum(fs.*coeffs2));
%             der_ests.V2{dd}(jj,ii) = der_ests.V2{dd}(ii,jj);
%         end
    end
%     der_ests.V2{dd}(abs(der_ests.V2{dd})<1e-5) = 0;
end
    
der_ests.T = cell(4,1);
for dd = 1:4
    der_ests.T{dd} = nan(size(Ts{dd}));
    der_ests.T2{dd} = nan(numel(Ts{dd}),numel(Ts{dd}));
    % coeffs = [-1 1];
    % dxs    = [0 1 ]*dx;
    for ii = 1:numel(Ts{dd})
        lls    = nan(numel(coeffs),1);
        for jj = 1:numel(coeffs)
            T1_c = Ts{dd};
            T1_c(ii) = T1_c(ii) + dxs(jj);
            
            Ts_c = Ts;
            Ts_c{dd} = T1_c;
            
            [~,~,lls(jj)] = gmlmRatesForTest(W,Ts_c,Us,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
        end
        der_ests.T{dd}(ii) = (coeffs*lls)./dx;
        
%         for jj = ii:numel(Ts{dd})
%             fs = zeros(length(dxs2),length(dxs2));
% 
%             for kk = 1:length(dxs2)
%                 for ll = 1:length(dxs2)
%                     if(coeffs2(kk,ll) ~= 0)
%                         xx_c = Ts;
%                         xx_c{dd}(ii) = xx_c{dd}(ii) + dxs2(kk);
%                         xx_c{dd}(jj) = xx_c{dd}(jj) + dxs2(ll);
%                         [~,~,fs(kk,ll)] = gmlmRatesForTest(W,xx_c,Us,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
%                     else
%                         fs(kk,ll) = 0;
%                     end
%                 end
%             end
% 
%             der_ests.T2{dd}(ii,jj) = sum(sum(fs.*coeffs2));
%             der_ests.T2{dd}(jj,ii) = der_ests.T2{dd}(ii,jj);
%         end
    end
%     der_ests.T2{dd}(abs(der_ests.T2{dd})<1e-5) = 0;
end

der_ests.U = cell(2,1);
for dd = 1:2
    der_ests.U{dd} = nan(size(Us{dd}));
    der_ests.U2{dd} = nan(numel(Us{dd}),numel(Us{dd}));
    % coeffs = [-1 1];
    % dxs    = [0 1 ]*dx;
    for ii = 1:numel(Us{dd})
        lls    = nan(numel(coeffs),1);
        for jj = 1:numel(coeffs)
            U1_c = Us{dd};
            U1_c(ii) = U1_c(ii) + dxs(jj);
            Us_c = Us;
            Us_c{dd} = U1_c;
            [~,~,lls(jj)] = gmlmRatesForTest(W,Ts,Us_c,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
        end
        der_ests.U{dd}(ii) = (coeffs*lls)./dx;
        
        
%         for jj = ii:numel(Us{dd})
%             fs = zeros(length(dxs2),length(dxs2));
% 
%             for kk = 1:length(dxs2)
%                 for ll = 1:length(dxs2)
%                     if(coeffs2(kk,ll) ~= 0)
%                         xx_c = Us;
%                         xx_c{dd}(ii) = xx_c{dd}(ii) + dxs2(kk);
%                         xx_c{dd}(jj) = xx_c{dd}(jj) + dxs2(ll);
%                         [~,~,fs(kk,ll)] = gmlmRatesForTest(W,Ts,xx_c,Vs,Fs,Gs,As,Bs,M_idx,T_idx, N, Y,dt);
%                     else
%                         fs(kk,ll) = 0;
%                     end
%                 end
%             end
% 
%             der_ests.U2{dd}(ii,jj) = sum(sum(fs.*coeffs2));
%             der_ests.U2{dd}(jj,ii) = der_ests.U2{dd}(ii,jj);
%         end
    end
    
%     der_ests.U2{dd}(abs(der_ests.U2{dd})<1e-5) = 0;
end