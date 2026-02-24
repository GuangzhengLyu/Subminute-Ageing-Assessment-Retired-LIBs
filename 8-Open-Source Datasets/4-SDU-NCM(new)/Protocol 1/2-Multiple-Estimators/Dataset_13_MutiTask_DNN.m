clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Build normalized multi-task targets (capacity, life, and
%%% expanded health indicators), normalize relaxation-based input features,
%%% run repeated leave-one-out cross-validation using a deep neural network
%%% (trainNetwork), visualize train/test predictions, and save Y_Test for
%%% downstream RMSE and consistency evaluation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_SDU_NCM_P1.mat')  
load('../Feature_ALL_SDU_NCM_P1.mat')

% Pack fitted relaxation parameters into a 3-D feature tensor
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1)    = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1)= OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Normalize to health-indicator scale (reference-based normalization)
Capa      = Capa/2.4;
Life      = Life;
ERate     = ERate/0.95;
CoChRate  = CoChRate/0.9;
MindVolt  = (MindVolt-2.65)/(3.65-2.65);
PlatfCapa = PlatfCapa/1.8;

%% Output normalization (min-max to [0, 1])
Max_Out = [1.09, 550, 0.996, 1.028, 1.05,  0.9];
Min_Out = [1.01, 400, 0.987, 1.008, 1.015, 0.87];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Expanded health indicators (normalized)
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip normalized outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (min-max to [0, 1])
    Max_In = [4.29, 0.5, 0.2,  1400, 0.1,  1400];
    Min_In = [4.19, -2,  0.04, 500,  0.01, 800];

    % Mapping from CountSV to the (intended) index used in Min_In/Max_In
    MyInd = 15-CountSV;
          
    % Normalize each input feature dimension for the selected relaxation segment (j=1)
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized inputs/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    %% Repeated leave-one-out cross-validation (LOOCV)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Copy full dataset and extract the current test sample
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Test set (one sample)
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Training set (remaining samples)
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Keep explicit reshape steps (as in original script)
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% Deep neural network (regression)
            % Define a lightweight fully-connected regression network
            layers = [
                sequenceInputLayer(6)
                fullyConnectedLayer(20)
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];
            
            % Training options (Adam optimizer)
            options = trainingOptions(...
                'adam', ...
                'MaxEpochs',           1000, ...
                'MiniBatchSize',       16, ...
                'InitialLearnRate',    0.001, ...
                'GradientThreshold',   0.9, ...
                'LearnRateSchedule',   'piecewise', ...
                'LearnRateDropPeriod', 10, ...
                'LearnRateDropFactor', 0.9, ...
                'Verbose',             0, ...
                'Plots',               'none'...
                );

            % Train network and predict
            net = trainNetwork(In_Train,Out_Train,layers,options);

            Y_Train = predict(net,In_Train);
            Y_Test(CountRP,:,IndexData) = predict(net,In_Test(:,IndexData));
        
            toc
        end

        % Clip predictions to [0, 1] before inverse normalization
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% Inverse normalization (back to original indicator scales)
        Y_Train(1,:) = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Train(1,:) = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Y_Train(2,:) = Y_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        Out_Train(2,:) = Out_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        
        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end
        
        %% Visualization: train vs prediction, test vs prediction (per output dimension)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save predictions for the current CountSV setting
    currentFile = sprintf('Result_13_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end