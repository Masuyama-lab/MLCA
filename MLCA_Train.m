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
function net = MLCA_Train(DATA, LABEL, net)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
CountLabel = net.CountLabel;     % counter for labels of each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
total_inputs = net.total_inputs; % the total number of inputs during node learning
temp_Ci = net.temp_Ci;           % counter for the number of training instances w/ label
temp_NCi = net.temp_NCi;         % counter for the number of training instances w/o label

Smooth = net.Smooth;             % Laplace smoothing
neibhbors = net.refNode;           % the number of reference nodes
Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM
minCIM = net.minCIM;             % similarity threshold


if size(weight) == 0
    CountLabel(1,:) = zeros(1, size(LABEL,2));
    temp_Ci = zeros(size(LABEL,2), neibhbors+1);
    temp_NCi = zeros(size(LABEL,2), neibhbors+1);
    likelihood = zeros(size(LABEL,2), neibhbors+1);
    likelihoodN = zeros(size(LABEL,2), neibhbors+1);
end


for sampleNum = 1:size(DATA,1)
    
    total_inputs = total_inputs + 1;
    
    % Compute a kernel bandwidth for CIM based on data points.
    if isempty(weight) == 1 || mod(sampleNum, Lambda) == 0
        estSig = SigmaEstimation(DATA, sampleNum, Lambda);
    end
    
    % Current data sample.
    input = DATA(sampleNum,:);
    label = LABEL(sampleNum,:);
    
    if size(weight,1) < 1 % In the case of the number of nodes in the entire space is small.
        % Add Node
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        CountNode(numNodes) = 1;
        adaptiveSig(numNodes) = estSig;
        CountLabel(numNodes,:) = label;
        
    else
        
        % Calculate CIM based on global mean adaptiveSig.
        globalCIM = CIM(input, weight, mean(adaptiveSig));
        gCIM = globalCIM;
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Lcim_s1, s1] = min(gCIM);
        gCIM(s1) = inf;
        [Lcim_s2, s2] = min(gCIM);
        
        if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, Lambda);
            CountLabel(numNodes,:) = label;
            
            neibhbors = net.refNode;
            if  numNodes < neibhbors
                neibhbors = numNodes;
            end
            
            % Compute a similarity between an input and nodes
            new_globalCIM = [globalCIM, 0];
            [~, orderCIM] = sort(new_globalCIM, 'ascend');
            index = orderCIM(1:neibhbors);
            
            % Update temp_Ci and temp_NCi, then compute a likelihood
            [likelihood,likelihoodN,temp_Ci,temp_NCi] = UpdateLikelihood(temp_Ci,temp_NCi,label,CountLabel,neibhbors,net.refNode,index,Smooth);
            
        else % Case 2 i.e., V >= CIM_k1
            CountNode(s1) = CountNode(s1) + 1;
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            CountLabel(s1,:) = CountLabel(s1,:) + label;
            
            if minCIM >= Lcim_s2 % Case 3 i.e., V >= CIM_k2
                % Update weight of s2 node.
                weight(s2,:) = weight(s2,:) + (1/(10*CountNode(s2))) * (input - weight(s2,:));
            end
            
            neibhbors = net.refNode;
            if  numNodes < neibhbors
                neibhbors = numNodes;
            end
            
            % Compute a similarity between an input and nodes
            [~, orderCIM] = sort(globalCIM, 'ascend');
            index = orderCIM(1:neibhbors);
            
            % Update temp_Ci and temp_NCi, then compute a likelihood
            [likelihood,likelihoodN,temp_Ci,temp_NCi] = UpdateLikelihood(temp_Ci,temp_NCi,label,CountLabel,neibhbors,net.refNode,index,Smooth);
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    % Compute Prior and PriorN probabilities
    for i=1:size(label,2)
        temp_Ci_P = sum(CountLabel(:,i));
        Prior(i,1) = (Smooth+temp_Ci_P)/(Smooth*2+total_inputs);
        PriorN(i,1) = 1-Prior(i,1);
    end
    
end % for sampleNum = 1:size(DATA,1)

net.numNodes = numNodes;      % Number of nodes
net.weight = weight;          % Mean of nodes
net.CountNode = CountNode;    % Counter for each node
net.adaptiveSig = adaptiveSig;
net.Lambda = Lambda;
net.CountLabel = CountLabel;
net.total_inputs = total_inputs;

net.Prior = Prior;
net.PriorN = PriorN;
net.temp_Ci = temp_Ci;
net.temp_NCi = temp_NCi;
net.likelihood = likelihood;
net.likelihoodN = likelihoodN;

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimation(DATA, sampleNum, Lambda)

if size(DATA,1) < Lambda
    exNodes = DATA;
elseif (sampleNum - Lambda) <= 0
    exNodes = DATA(1:Lambda,:);
elseif (sampleNum - Lambda) > 0
    exNodes = DATA( (sampleNum+1)-Lambda:sampleNum, :);
end

% Scaling [0,1]
% normalized = (exNodes-min(exNodes))./(max(exNodes)-min(exNodes));
% qStd = std(normalized);
% qStd(isnan(qStd))=0;

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end

% Update temp_Ci and temp_NCi, then compute a likelihood
function [likelihood,likelihoodN,temp_Ci,temp_NCi] = UpdateLikelihood(temp_Ci,temp_NCi,label,CountLabel,neighbors,refNode,index,Smooth)

temp = sum( CountLabel(index(1:neighbors),:) );
temp = round((temp./max(temp)) * neighbors);
temp(isnan(temp)) = 0;

% Update temp_Ci, temp_NCi
if neighbors > 1
    for i = 1:size(label,2)
        if label(1,i) == 1
            temp_Ci(i,temp(i)+1) = temp_Ci(i,temp(i)+1) + 1;
        else
            temp_NCi(i,temp(i)+1) = temp_NCi(i,temp(i)+1) + 1;
        end
    end
else
    for i = 1:size(label,2)
        if label(1,i) == 1
            temp_Ci(i,refNode+1) = temp_Ci(i,refNode+1) + 1;
        else
            temp_NCi(i,refNode+1) = temp_NCi(i,refNode+1) + 1;
        end
    end
end

% likelihood
likelihood = zeros(size(temp_Ci));
likelihoodN = zeros(size(temp_NCi));
for i = 1:size(label,2)
    temp1 = sum(temp_Ci(i,:));
    temp2 = sum(temp_NCi(i,:));
    for j = 1:neighbors+1
        likelihood(i,j) = (Smooth + temp_Ci(i,j)) / (Smooth * (neighbors+1) + temp1); 
        likelihoodN(i,j) = (Smooth + temp_NCi(i,j)) / (Smooth * (neighbors+1) + temp2);
    end
end

end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

% ret0 = GaussKernel(0, sig);
ret0 = 1;
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end



