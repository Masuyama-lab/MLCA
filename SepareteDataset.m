function [Xtrain, Ytrain, Xtest, Ytest] = SepareteDataset(data,target,indices,trial,fold)


data = sparse(data);
index = indices(:,trial);

test = (index == fold);
train = ~test;
Xtrain = data(train,:);
Ytrain = target(train,:);
Xtest = data(test,:);
Ytest = target(test,:);

rng(fold);
ran = randperm(size(Xtrain,1));
Xtrain = Xtrain(ran,:);
Ytrain = Ytrain(ran,:);

end