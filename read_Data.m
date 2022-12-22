
function [data,target,indices]=read_Data(dataname,numCV,TRIAL)
%% Input
% dataname: name of dataset (call dataname.mat from dataset/matfile/
% numCV   : the number of separation of data
%% Output
% data    : feature matrix (N x F matrix)
% target  : label   matrix (L x N matrix) 
% indices : assign vectors (N x 10 matrix  in [numCV])
%% Options
%seed    :  seed for random vairables
%Without seed, we load preseparated indices from dataset/index
%             (recommended for comparisons)
%With seed,  we separate dataset to train/test with the given seed 

% load matrix files
tmp=load([dataname,'.mat']);
data=tmp.data;
target=tmp.target;
target(target == -1) = 0;

indices=[];
for i=1:TRIAL
    rng(i);
    ind = datasplitind( size(data,1), numCV );
    
    data=sparse(data);
    indices=[indices ind];
end

end


function [ ind ] = datasplitind( datasize, kfold )
subSizes = diff(round(linspace(0, datasize, kfold+1)));
ind = repelem(1:kfold, subSizes(randperm(kfold)))';
% randomizing the output regions
ind = ind(randperm(datasize));
end
