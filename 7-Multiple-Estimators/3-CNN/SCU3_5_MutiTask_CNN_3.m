clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Train and evaluate a CNN estimator on SCU3 Dataset #3.
%%% It builds six targets (SOH, RUL proxy, and four expanded indicators),
%%% applies per-setpoint min–max normalization for inputs and outputs, runs
%%% repeated leave-one-out testing across samples, and saves Y_Test for each
%%% terminal-voltage setpoint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
load('../../OneCycle_3.mat')
load('../../Feature_3_ALL.mat')

% Assemble feature tensor: [sample, setpoint, feature_dim]
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 1.75 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH (scaled to nominal reference)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Unified health-indicator scaling
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization (task-wise min/max)
Max_Out = [0.72, 1800, 0.9,  0.8,  0.75, 0.4];
Min_Out = [0.57, 400,  0.74, 0.15, 0,    0];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip normalized outputs to [0, 1]
Output(Output<0) = 0; Output(Output>1) = 1;

%% CNN (per terminal-voltage setpoint)
for CountSV = 13:-1:1

    %% Input normalization (setpoint-wise min/max)
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

    % Map CountSV (13..1) to row index (1..13) in min/max tables
    MyInd = 14-CountSV;

    % Normalize features at the selected setpoint
    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,CountSV,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,CountSV,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clip normalized inputs/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0; Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0; Output(Output>1) = 1;

    %% Repeated leave-one-out evaluation
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Copy full set and create one-sample test split
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % CNN input formatting:
            % In_Test has shape [numChannels, 6, 1, 1]
            for NumST = 1:size(Temp_F,3)
                In_Test(NumST,:,1,1) = Temp_F(IndexData,:,NumST);
            end
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove test sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Build training tensor: [numChannels, 6, 1, numTrainSamples]
            for NumST = 1:size(Temp_F,3)
                for NumID = 1:size(Temp_F,1)
                    In_Train(NumST,:,1,NumID) = Temp_F(NumID,:,NumST);
                end
            end
            Out_Train = Temp_O';

            %% CNN definition (simple conv + FC regression head)
            layers = [
                imageInputLayer([size(In_Train,1) 6 1])
                convolution2dLayer([ceil(size(In_Train,1)/2),3],20,'Padding','same')
                maxPooling2dLayer([1,1],'Stride',2)
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];

            % Training options
            options = trainingOptions(...
                'adam', ...
                'MaxEpochs',           200, ...
                'MiniBatchSize',       32, ...
                'InitialLearnRate',    0.01, ...
                'GradientThreshold',   0.9, ...
                'LearnRateSchedule',   'piecewise', ...
                'LearnRateDropPeriod', 5, ...
                'LearnRateDropFactor', 0.9, ...
                'Verbose',             0, ...
                'Plots',               'none' ...
                );

            % Train and predict
            net = trainNetwork(In_Train,Out_Train',layers,options);
            Y_Train = predict(net,In_Train)';                  % [6, numTrainSamples]
            Y_Test(CountRP,:,IndexData) = predict(net,In_Test); % [1, 6]

            toc
        end

        % Clip predictions to [0, 1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0; Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% De-normalize to physical scales
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

        %% Visualization (train scatter and test scatter)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save per-setpoint test predictions
    currentFile = sprintf('CNN_Result_3_50_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end