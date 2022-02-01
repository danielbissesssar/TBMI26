function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);
for i = 1:NClasses
   idx = LTrue == i;
   for j = 1:NClasses
       cM(i,j) = sum(LPred(idx) == j);
   end   
end
end

