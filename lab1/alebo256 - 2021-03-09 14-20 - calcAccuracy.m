function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% Add your own code here
%acc = 0;
d = sum(diag(cM));
t = sum(cM, 'all');
acc = d/t;
end

