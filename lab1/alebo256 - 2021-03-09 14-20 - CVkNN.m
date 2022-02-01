% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );
%plotCase(X,D)
numBins = 5;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
perfs = zeros(30,1);
for k = 1:30
    acc = 0;
    for n = 1:numBins
        T = [];
        L = [];
        for i = 1:numBins
            if (i~=n)
                T = [T; XBins{i}];
                L = [L; LBins{i}];
            end    
        end
        LPredTest  = kNN(XBins{n} , k, T, L);
        cM = calcConfusionMatrix(LPredTest, LBins{n});
        acc = acc + calcAccuracy(cM);
    end
    perfs(k) = acc/numBins;
end
%[~,I] = maxk(accuracy, );
%bestK = max(I)
scatter(1:30, perfs)
