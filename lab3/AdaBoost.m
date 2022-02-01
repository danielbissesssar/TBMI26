%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 250;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 90;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

threshold = 0;
bestX = 0;
bestP = 0;
bestAll = zeros(4, nbrWeakClassifiers);
D = ones(1, size(xTrain,2))/size(xTrain, 2);
for i = 1:nbrWeakClassifiers
    Emin = inf;
    for j = 1:size(xTrain,1)
        for k = 1:size(xTrain,2)
            P = 1;
            C = WeakClassifier(xTrain(j,k), P, xTrain(j,:)');
            E = WeakClassifierError(C, D, yTrain);
            if (E > 0.5)
                P = P*-1;
                E = 1-E;
            end
            if (E < Emin)
                Emin = E;
                threshold = xTrain(j,k);
                bestC = C*P;
                bestX = j;
                bestP = P;
            end
        end
    end
    a = log((1-Emin)/Emin)/2;
    bestAll(:,i) = [bestX; threshold; bestP; a];
    D = D.*exp(-a*yTrain.*bestC');
    D = D/sum(D);
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.
res = zeros(length(xTest),1);
accTest = zeros(size(bestAll,2),1);
for i = 1:size(bestAll,2)
    weak = WeakClassifier(bestAll(2,i), bestAll(3,i), xTest(bestAll(1,i),:)');
    Hi = bestAll(4,i)*weak;
    res = res + Hi;
    accTest(i,1) = 1-sum(sign(res)' ~= yTest)/length(yTest);
end    
class = sign(res);

res = zeros(length(xTrain),1);
accTrain = zeros(size(bestAll,2),1);
for i = 1:size(bestAll,2)
    weak = WeakClassifier(bestAll(2,i), bestAll(3,i), xTrain(bestAll(1,i),:)');
    Hi = bestAll(4,i)*weak;
    res = res + Hi;
    accTrain(i,1) = 1-sum(sign(res)' ~= yTrain)/length(yTrain);
end    
classTrain = sign(res);


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure();
plot(1:nbrWeakClassifiers, accTest);
hold on;
plot(1:nbrWeakClassifiers, accTrain);
legend("Test", "Train")

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.
diff = class'- yTest;
ind = find(diff);
figure();
colormap gray;
for t = 1:25
 subplot(5,5,t);
imagesc(testImages(:,:,ind(t)))   
end
figure();
colormap gray;
i = 1;
for t = 750:774 
    subplot(5,5,i);
    imagesc(testImages(:,:,ind(t)))
    i = i+1;
end
%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
figure();
colormap gray;
subplot(331)
imagesc(haarFeatureMasks(:,:,1))
subplot(332)
imagesc(haarFeatureMasks(:,:,10))
subplot(333)
imagesc(haarFeatureMasks(:,:,20))
subplot(334)
imagesc(haarFeatureMasks(:,:,30))
subplot(335)
imagesc(haarFeatureMasks(:,:,40))
subplot(336)
imagesc(haarFeatureMasks(:,:,50))
subplot(337)
imagesc(haarFeatureMasks(:,:,60))
subplot(338)
imagesc(haarFeatureMasks(:,:,69))

