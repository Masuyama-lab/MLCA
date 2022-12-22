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
function [conf,pred] = MLCA_Test(Xt, net, threshold)

weight = net.weight;             % node position
CountLabel = net.CountLabel;     % counter for labels of each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
Prior = net.Prior;               % prior probabilities
PriorN = net.PriorN;             % PriorN = 1-Prior
likelihood = net.likelihood;     % likelihood
likelihoodN = net.likelihoodN;
% refNode = net.refNode;           % the number of reference nodes


[num_testing, ~] = size(Xt);
[~, num_class] = size(CountLabel);
outputs = zeros(num_class, num_testing);

% Classify test data by disjoint clusters
for sampleNum = 1:size(Xt,1)
    
    % Current data sample
    input = Xt(sampleNum,:);
    
    % Compute a similarity between an input and nodes
    clusterCIM = CIM(input, weight, median(adaptiveSig));
    [~, orderCIM] = sort(clusterCIM, 'ascend');
    numNodes = net.numNodes;
    refNode = net.refNode;
    if  numNodes < refNode
        refNode = numNodes;
    end
    index = orderCIM(1:refNode);
    temp = sum( CountLabel(index(1:refNode),:),1 );
    temp = round((temp./max(temp)) * refNode);
    temp(isnan(temp)) = 0;
    
    % Compute label probability for test data
    for i=1:num_class
        Prob_in = likelihood(i, temp(1,i)+1) * Prior(i);
        Prob_out = likelihoodN(i, temp(1,i)+1) * PriorN(i);
        if(Prob_in+Prob_out == 0)
            outputs(i,sampleNum) = Prior(i);
        else
            outputs(i,sampleNum) = Prob_in/(Prob_in+Prob_out);
        end
    end
    
    
end


% Thresholding
conf = outputs';

for k = 1:size(conf,1)
%     conf(k,:) = normalize(conf(k,:),'range');
end

% Normalization [0-1]
% for k=1:size(conf,1)
%     mmin = min(conf(k,:));
%     mmax = max(conf(k,:));
%     conf(k,:) = (conf(k,:)-mmin) ./ (mmax-mmin);
% end
% conf(isnan(conf))=0;


pred = conf;
pred(pred > threshold) = 1;
pred(pred <= threshold) = 0;

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

ret0 = GaussKernel(0, sig);
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end



