%%This code actually works
%%This code is a neural network that works to determine the difference
%%between cracks, inclusions, and matrices in images

categories = {'Cracks', 'Inclusions'};
rootFolder = 'C:\Users\nbtar\Documents\MSE 490\Training Data';
imds = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

%%The following architecture is defining layers
varSize = 32;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(2,'BiasLearnRateFactor',2);   %2 = number of classes we are analyzing
fc2.Weights = gpuArray(single(randn([2 64])*0.1));      %2 = number of classe we are analyzing

layers = [
    imageInputLayer([varSize varSize 3]);               %imageInputLayer inputs 2-D images to a network and applies data normalization
    
    conv1;                                              %from the convl function above
    batchNormalizationLayer;                            %normalizes the activations and gradients in the network- optimizes the training
    reluLayer();                                        %nonlinear activation function
    
    maxPooling2dLayer(3,'Stride',2);                    %down-sampling operation 
    reluLayer();
    
    convolution2dLayer(5,32,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
   
    averagePooling2dLayer(3,'Stride',2);    
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    
    fc2;
    softmaxLayer()
    classificationLayer()];

%%Defining training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%%One line code to train the network
[net, info] = trainNetwork(imds, layers, options);

%%Loading test data
rootFolder = 'C:\Users\nbtar\Documents\MSE 490\Testing Data';
imds_test = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

labels = classify(net, imds_test);
ii = randi(50);
im = imread(imds_test.Files{ii});
imshow(im);
if labels(ii) == imds_test.Labels(ii)
   colorText = 'g'; 
else
    colorText = 'r';
end
title(char(labels(ii)),'Color',colorText);

%%This could take a while if you are not using a GPU
confMat = confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

%run the network on the test set (different from train set) and predict the
%image labels
YPred = classify(net,imds_test);
YTest = imds_test.Labels;


