clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Train and evaluate a DNN on SCU3 Dataset #3 (terminal-voltage sweep).
%%% It constructs six targets (SOH, life/RUL proxy, and four expanded indicators). 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
load('../../OneCycle_3.mat')
load('../../Feature_3_ALL.mat')

% Assemble feature tensor: Feature(sample, setpoint, feature_dim)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Build samples and targets (ground truth)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData + 1;

        % Define life as the first cycle where discharge capacity drops below 1.75 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Raw capacity and expanded indicators from the selected cycle index
        Capa(CountData,1)      = OneCycle(IndexData).OrigCapaAh;
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Convert to normalized health indicators (engineering definitions)
Capa      = Capa/3.5;
Life      = Life;
ERate     = ERate/89;
CoChRate  = CoChRate/83;
MindVolt  = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output min–max normalization bounds (dataset-specific)
Max_Out = [0.72, 1800, 0.9,  0.8,  0.75, 0.4];
Min_Out = [0.57, 400,  0.74, 0.15, 0,    0];

% Pack multi-task outputs: Output(sample, task)
Output(:,1) = (Capa      - Min_Out(1)) / (Max_Out(1) - Min_Out(1));
Output(:,2) = (Life      - Min_Out(2)) / (Max_Out(2) - Min_Out(2));
Output(:,3) = (ERate     - Min_Out(3)) / (Max_Out(3) - Min_Out(3));
Output(:,4) = (CoChRate  - Min_Out(4)) / (Max_Out(4) - Min_Out(4));
Output(:,5) = (MindVolt  - Min_Out(5)) / (Max_Out(5) - Min_Out(5));
Output(:,6) = (PlatfCapa - Min_Out(6)) / (Max_Out(6) - Min_Out(6));

% Clip outputs to [0,1] after normalization
Output(Output<0) = 0;
Output(Output>1) = 1;

%% DNN training/evaluation across terminal-voltage setpoints
for CountSV = 13:-1:1

    %% Input min–max normalization bounds for each terminal-voltage setpoint
    Max_In = [4,    0.16, 0.085, 650, 0.05,  700;
              3.92, 0.18, 0.1,   700, 0.04,  750;
              3.83, 0.17, 0.1,   700, 0.05,  800;
              3.73, 0.18, 0.09,  750, 0.05,  900;
              3.63, 0.18, 0.09,  700, 0.05,  900;
              3.5,  0.17, 0.085, 650, 0.055, 850;
              3.37, 0.17, 0.072, 550, 0.055, 600;
              3.28, 0.17, 0.062, 550, 0.05,  350;
              3.18, 0.17, 0.05,  600, 0.05,  300;
              3.08, 0.155,0.04,  600, 0.04,  450;
              2.99, 0.142,0.034, 600, 0.03,  400;
              2.94, 0.13, 0.035, 450, 0.02,  280;
              2.9,  0.11, 0.02,  280, 0.02,  280];

    Min_In = [3.7,  0.05, 0.04,  350, 0.005, 0;
              3.6,  0.04, 0.04,  250, 0.005, 0;
              3.48, 0.05, 0.04,  200, 0.005, 0;
              3.38, 0.05, 0.04,  200, 0.004, 0;
              3.28, 0.05, 0.04,  200, 0.004, 0;
              3.2,  0.06, 0.045, 200, 0.005, 0;
              3.13, 0.07, 0.052, 200, 0.005, 50;
              3.05, 0.07, 0.042, 200, 0.01,  50;
              2.99, 0.08, 0.033, 0,   0.01,  50;
              2.94, 0.08, 0.024, 50,  0.015, 50;
              2.89, 0.08, 0.014, 60,  0.006, 60;
              2.84, 0.04, 0.008, 70,  0.004, 90;
              2.78, 0.02, 0.004, 100, 0.004, 110];

    % Map CountSV (13..1) to row index (1..13)
    MyInd = 14 - CountSV;

    % Normalize features at this setpoint into [0,1]
    Feature_Nor(:,1) = (Feature(:,CountSV,1) - Min_In(MyInd,1)) / (Max_In(MyInd,1) - Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2) - Min_In(MyInd,2)) / (Max_In(MyInd,2) - Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3) - Min_In(MyInd,3)) / (Max_In(MyInd,3) - Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4) - Min_In(MyInd,4)) / (Max_In(MyInd,4) - Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,CountSV,5) - Min_In(MyInd,5)) / (Max_In(MyInd,5) - Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,CountSV,6) - Min_In(MyInd,6)) / (Max_In(MyInd,6) - Min_In(MyInd,6));

    % Clip normalized inputs/outputs to [0,1]
    Feature_Nor(Feature_Nor<0) = 0;
    Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;
    Output(Output>1) = 1;

    %% Repeat leave-one-out evaluation 100 times
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Copy full dataset, then hold out one sample as test
            Temp_F = Feature_Nor;
            Temp_O = Output;

            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Channel-based exclusion to reduce leakage for RUL-related evaluation:
            % remove all samples that share the same tester channel as the test sample
            CountSamChannel = 0;
            for IndDataCn = 1:length(Feature_Nor)
                if OneCycle(IndDataCn).Channel == OneCycle(IndexData).Channel
                    CountSamChannel = CountSamChannel + 1;
                    SamChannelIndx(CountSamChannel) = IndDataCn;
                end
            end
            Temp_F(SamChannelIndx,:) = [];
            Temp_O(SamChannelIndx,:) = [];

            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Ensure 2-D shapes
            In_Train = reshape(In_Train,[size(In_Train,1), size(In_Train,2)]);
            In_Test(:,IndexData) = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1), size(In_Test(:,IndexData),2)]);

            % Define a compact fully-connected regressor (multi-output)
            layers = [
                sequenceInputLayer(6)
                fullyConnectedLayer(20)
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];

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
                'Plots',               'none' ...
                );

            % Train and predict
            net = trainNetwork(In_Train, Out_Train, layers, options);
            Y_Train = predict(net, In_Train);
            Y_Test(CountRP,:,IndexData) = predict(net, In_Test(:,IndexData));

            toc
        end

        % Clip predictions in normalized space to [0,1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;
        Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        % De-normalize predictions and ground truth to physical units
        Y_Train(1,:) = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1)) + Min_Out(1));
        Out_Train(1,:) = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1)) + Min_Out(1));
        Y_Train(2,:) = Y_Train(2,:)*(Max_Out(2)-Min_Out(2)) + Min_Out(2);
        Out_Train(2,:) = Out_Train(2,:)*(Max_Out(2)-Min_Out(2)) + Min_Out(2);

        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1)) + Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1)) + Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp)) + Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp)) + Min_Out(IndOp);
        end

        % Visualize train/test scatter for quick sanity checking
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:), Y_Train(IndOTrain,:), 'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)), squeeze(Y_Test(CountRP,IndOTrain,:)), 'o')
        end

    end

    % Save predictions for this terminal-voltage setpoint
    currentFile = sprintf('DNN_Result_3_50_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end