clear mex

singlePrecision = true;
gmlm = GMLM(Y,M_idx,T_idx,dt,singlePrecision,@gmlmPrior_normalW,0);

if(singlePrecision)
    typeFunc = @single;
else
    typeFunc = @double;
end

dim1_a_normalized = A1;
dim1_b = B1;

dim2_a = A2;
dim2_b = B2;

dim3_a = A3;
dim4_a_orthonormal = zeros(size(A3,1),0);

Regressors.dim1_a_normalized = F1;
Regressors.dim1_b = G1;

Regressors.dim2_a = F2;
Regressors.dim2_b = G2;

Regressors.dim3_a = F3;

Regressors.dim4_a_orthonormal = F4;
Regressors.testSpace = randn(ceil(500e6/8),1);
RegressorIndex = table(dim1_a_normalized,dim1_b,dim2_a,dim2_b,dim3_a,dim4_a_orthonormal);

dimGroup1 = {'dim1_a_normalized','dim1_b'};
dimGroup2 = {'dim2_a','dim2_b'};
dimGroup3 = {'dim3_a'};
dimGroup4 = {'dim4_a_orthonormal'};

gmlm = gmlm.addGroup(R1,R1,[],0,'comp 1: 3-T',Regressors,RegressorIndex(:,dimGroup1));
gmlm = gmlm.addGroup(R2,R2,[],0,'comp 2: 3-T',Regressors,RegressorIndex(:,dimGroup2));
gmlm = gmlm.addGroup(R3,R3,[],0,'comp 3: 2-T',Regressors,RegressorIndex(:,dimGroup3));
gmlm = gmlm.addGroup(R4,R4,[],0,'comp 4: 2-T',Regressors,RegressorIndex(:,dimGroup4));

gmlm = gmlm.setGroupPrior(1,@gmlmPrior_normalVT,0);
gmlm = gmlm.setGroupPrior(2,@gmlmPrior_normalVT,0);
gmlm = gmlm.setGroupPrior(3,@gmlmPrior_normalVT,0);
gmlm = gmlm.setGroupPrior(4,@gmlmPrior_normalVT,0);


kcResetDevices(0);
gmlm = gmlm.toGPU(0);  %resObj = gmlm.computeLL(paramObj,optObj,resObj);

%%
paramStruct = gmlm.getEmptyParamStruct();
optsStruct = gmlm.getEmptyOptsStruct(true,true);
resultsStruct = gmlm.getEmptyResultsStruct();

paramStruct.Groups(1).T{1} = typeFunc(T1*mm1);
paramStruct.Groups(1).T{2} = typeFunc(U1*mm1);
paramStruct.Groups(1).V    = typeFunc(V1*mm1);

paramStruct.Groups(2).T{1} = typeFunc(T2*mm2);
paramStruct.Groups(2).T{2} = typeFunc(U2*mm2);
paramStruct.Groups(2).V(:) = typeFunc(V2*mm2);

paramStruct.Groups(3).T{1} = typeFunc(T3*mm3);
paramStruct.Groups(3).V(:) = typeFunc(V3*mm3);

paramStruct.Groups(4).T{1} = typeFunc(T4*mm4);
paramStruct.Groups(4).V(:) = typeFunc(V4*mm4);
paramStruct.W(:)           = typeFunc(W*mmw);




resultsStruct2 = gmlm.computeLL(paramStruct,optsStruct);
% close all
% figure(1);
% clf
% subplot(1,2,1);
% plot([lambda_0 lambda])
% subplot(1,2,2);
% plot([lambda_0-lambda])
% [ll_1,lambda_1] = gmlm.computeLL_cpu();

% get LL: on CPU

% get GPU LL & derivatives

% test derivatives with finite difference