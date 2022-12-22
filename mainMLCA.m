% 
% (c) 2022 Naoki Masuyama
% 
% These are the codes of Multi-label CIM-based Adaptive Resonance Theory (MLCA)
% proposed in N. Masuyama, Y. Nojima, C. K. Loo, and H. Ishibuchi,
% "Multi-label classification via adaptive resonance theory-based clustering,"
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
% 
% Please contact "masuyama@omu.ac.jp" if you have any problems.
%    


numCV = 10;   % for cross validation
TRIAL = 2;    % Number of trials


% Dataset
dataname = 'emotions';

% Set train and test data
[data,target,indices] = read_Data(dataname, numCV, TRIAL);

% for result
time_mlca_train = 0;
time_mlca_test = 0;
outputMLCA = zeros(19,numCV);
numNodesMLCA = zeros(1,numCV);
ResMLCA = zeros(20,numCV*TRIAL);


for trial = 1:TRIAL
    
    fprintf('Iterations: %d/%d\n',trial,TRIAL);
    
    
%     parfor fold = 1:numCV
    for fold = 1:numCV
        
        % Parameters of MLCA =================================================
        MLCAnet.numNodes    = 0;   % the number of nodes
        MLCAnet.weight      = [];  % node position
        MLCAnet.CountNode = [];    % winner counter for each node
        MLCAnet.CountLabel = [];   % counter for labels of each node
        MLCAnet.adaptiveSig = [];  % kernel bandwidth for CIM in each node
        MLCAnet.total_inputs = 0;  % the total number of inputs during node learning
        MLCAnet.temp_Ci = [];      % counter for the number of training instances w/ label
        MLCAnet.temp_NCi = [];     % counter for the number of training instances w/o label
        MLCAnet.Prior = [];        % prior probabilities
        MLCAnet.PriorN = [];       % PriorN = 1-Prior
        MLCAnet.likelihood = [];   % likelihood
        MLCAnet.likelihoodN = [];
        MLCAnet.Smooth = 1;        % Laplace smoothing
        
        MLCAnet.refNode = 10;      % the number of reference nodes
        MLCAnet.Lambda = 50;       % an interval for calculating a kernel bandwidth for CIM
        MLCAnet.minCIM = 0.55;      % similarity threshold
        % ====================================================================
        
        
        % Separate dataset into training and test for CV
        [X, Y, Xt, Yt] = SepareteDataset(data,target,indices,trial,fold);
        
        
        % Train
        tic
        MLCAnet = MLCA_Train(X, Y, MLCAnet);
        time_mlca_train = toc;
        
        % Test
        tic
        [confMLCA, predMLCA] = MLCA_Test(Xt, MLCAnet, 0.5);
        time_mlca_test = toc;
        
        % Evalution
        [outputMLCA(:,fold),~] = Evaluation(Yt,confMLCA,predMLCA,time_mlca_train,time_mlca_test,3);
        numNodesMLCA(:,fold) = MLCAnet.numNodes;
        
    end
    
    % result
    ResMLCA(:,1+(trial-1)*numCV:trial*numCV) = [outputMLCA; numNodesMLCA];

end



% averaged results
result_MLCA = squeeze(mean(ResMLCA,2));
traMLCA = ResMLCA';
result_std_MLCA = squeeze(std(traMLCA))';


%Visualization
% metListMLCA = {'top1','top3','top5','dcg1','dcg3','dcg5','auc','exact','hamming','macroF1','microF1','fscore','acc','pre','rankLoss','one','cov','trainT','testT','# Nodes'};
metListMLCA = {'auc','exact','hamming','macroF1','microF1','fscore','acc','pre','rankLoss','one','cov','trainT','testT','# Nodes'};
array2table([result_MLCA(7:end,:),result_std_MLCA(7:end,:)],'RowNames',metListMLCA,'VariableNames',{'Mean','Std'})

