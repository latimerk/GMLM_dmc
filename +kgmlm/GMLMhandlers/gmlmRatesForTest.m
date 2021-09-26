function [lambda,S,ll,ll_M,ll_T,ll_all] = gmlmRatesForTest(W,Ts,Us,Vs,Xs,Ys,As,Bs,M_idx,T_idx,N,S,dt)

M = length(M_idx);
T = length(T_idx);
M_idx = [M_idx;N+1];
T_idx = [T_idx;N+1];

C = numel(Xs);

if(C ~= numel(Ys) || C ~= numel(Bs) || C ~= numel(As) || C ~= numel(Ts) || C ~= numel(Us) || C ~= numel(Vs))
    error('inconsistent number of elements');
end

%setup constant value
lambda = zeros(N,1);
for ii = 1:M
    lambda(M_idx(ii):(M_idx(ii+1)-1)) = W(ii);
end

%for each component
for cc = 1:C
    X_c = Xs{cc};
    Y_c = Ys{cc};
    
    A_c = As{cc};
    B_c = Bs{cc};
    
    T_c = Ts{cc};
    U_c = Us{cc};
    V_c = Vs{cc};
    
    if(isempty(A_c))
        if(mod(size(X_c,1),N) == 0 && size(X_c,1) > 0)
            D = size(X_c,1)/N;
        else
            error('Incorrect dims for X %d',cc);
        end
    else
        D = size(A_c,2);
    end
    
    %if component is 3-Tensor
    if(~isempty(Y_c))
        useY = true;
        
        if(isempty(B_c))
            if(mod(size(Y_c,1),N) == 0 && size(Y_c,1) > 0)
                Db = size(Y_c,1)/N;
            else
                error('Incorrect dims for Y %d',cc);
            end
        else
            Db = size(B_c,2);
        end
        
        if(Db ~= D)
            error('Index dims for X and Y %d do not match',cc);
        end
    else
        useY = false;
    end
    
    %for each additive part of the component
    R_c = size(T_c,2);
    for dd = 1:D
        
        if(~isempty(A_c))
            XT = zeros(N,R_c);
            vi = A_c(:,dd) >= 1 & A_c(:,dd) <= N;
            XT(vi,:) = X_c(A_c(vi,dd),:)*T_c;
        else
            XT = X_c((1:N)+(dd-1)*N,:)*T_c;
        end
        
        
        if(useY)
            if(~isempty(B_c))
            	YU = zeros(N,R_c);
                vi = B_c(:,dd) >= 1 & B_c(:,dd) <= N;
                YU(vi,:) = Y_c(B_c(vi,dd),:)*U_c;
            else
                YU = Y_c((1:N)+(dd-1)*N,:)*U_c;
            end
        else
            YU = 1;
        end
        
        XtYu = XT.*YU;
        
        %for each neuron
        for ii = 1:M
            idx = M_idx(ii):(M_idx(ii+1)-1);
            lambda(idx) = lambda(idx) + sum(XtYu(idx,:).*V_c(ii,:),2);
        end
    end
end

%%
lambda = lambda + log(dt);

if(nargout > 1 && (nargin < 12 || isempty(S)))
    S = poissrnd(exp(lambda));
end

ll_all = -exp(lambda(:)) + S(:).*lambda(:) - gammaln(S(:)+1);

ll = sum(ll_all);

ll_M = zeros(M,1);
for ii = 1:M
    idx = M_idx(ii):(M_idx(ii+1)-1);
    ll_M(ii) = sum(ll_all(idx));
end

ll_T = zeros(T,1);
for ii = 1:T
    idx = T_idx(ii):(T_idx(ii+1)-1);
    ll_T(ii) = sum(ll_all(idx));
end
