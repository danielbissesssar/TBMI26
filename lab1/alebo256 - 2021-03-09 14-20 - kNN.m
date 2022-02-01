function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);
dist = pdist2(X, XTrain);
[~, ind] = mink(dist, k, 2);
labels = zeros(size(ind,1),k);
for i = 1:size(ind,1)
    labels(i,:) = LTrain(ind(i,:));
end 
LPred = mode(labels, 2);  
end

