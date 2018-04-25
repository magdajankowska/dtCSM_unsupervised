function [mappedData,mapping]=dtCSM_unsup(train, architecture, DR_DIM,iter,hDimDist,realData,batchSize, outFileStandarization,outFilePretrainNetwork)

% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author(s).

% (C) Ryan Kiros, 2012
% Dalhousie University 
% (C) Axel Soto, 2013
% Dalhousie University 

%----------------------------------------------------
%INPUT
%train: instances x features matrix of training data
%architecture: number of nodes per layer. e.g. [500,100,50,2] (including last layer)
%DR_DIM: output dimensionality
%iter: number of iterations for finetuning
%hDimDist: distance used for high-dimensional instances
%realData (0/1) set to 1 if data values are real
%batchSize: size for batch
%outFileStandarization,outFilePretrainNetwork (optional) - names of files to which save, respectively
%                                                          data related to standarization and pretrained network
%                                                          if not defined, the names "values_for_standarization.mat" and
%                                                          "network2.mat" will be used by default
%                                                          data related to standarization (saved in outFileStandarization) consists of three matrices:
%                                                          - meanTrain is the mean of each train data column 
%                                                               (that was subtracted from input data before training)
%                                                          - stdTrain is the standard deviation of train data column 
%                                                               (by which each column of input data was subsequently divided)
%                                                          - zeroColumns are indices of columns in train data that have zero standard deviation
%                                                               (these columns have values replaced by zeros in the input data)
%                                                           pretrained network (saved in outFilePretrainNetwork) is a cell array named network2
%----------------------------------------------------
%OUTPUT
%mappedData: matrix of mapped data (instances x DR_DIM)
%mapping: trained weights for the network

cd tCMM

%Pretrain the network
if (realData)
    %Standardize the data
    stdTrain=std(train);
    meanTrain=mean(train);
    zeroColumns=find(stdTrain<=eps);
    
    train= bsxfun(@minus, train, meanTrain);
    train = bsxfun(@rdivide, train, stdTrain);
    train(:,zeroColumns)=0; 
    
    %if values for standarizations need to be used later
    %for other data
    if ~exist('outFileStandarization','var') || isempty(outFileStandarization)
        outFileStandarization='values_for_standarization.mat'   
    end
    save(outFileStandarization,'stdTrain','meanTrain','zeroColumns');


    network2 = tcmm(train, architecture, 'CD1', 1);
else
    network2 = tcmm(train, architecture, 'CD1', 0);
end
    

%If pretraining needs to be reused
if ~exist('outFilePretrainNetwork','var') || isempty(outFilePretrainNetwork)
    outFilePretrainNetwork='network2.mat';
end

save(outFilePretrainNetwork,'network2');
%load('network2.mat');

% Fine-tune all data
network_f = cmm_r_backprop_selfDistsRemoved(network2, train,  ...
    iter, DR_DIM - 1, DR_DIM, hDimDist, batchSize);
    
% Run the data through the network
mappedData = run_data_through_network(network_f, train);
mapping=network_f;

cd ..
