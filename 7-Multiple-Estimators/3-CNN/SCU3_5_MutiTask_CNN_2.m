clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Train and evaluate a CNN estimator on SCU3 Dataset #2.
%%% It constructs six ageing/health targets (SOH and expanded indicators),
%%% normalizes inputs/outputs per terminal-voltage setpoint, runs repeated
%%% leave-one-out testing across samples, and saves predictions (Y_Test)
%%% for each setpoint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
load('../../OneCycle_2.mat')
load('../../Feature_2_ALL.mat')

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

        % Define life as the first cycle where discharge capacity drops below 2.1 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
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
Max_Out = [0.716971, 805, 0.9,  0.83, 0.75, 0.356];
Min_Out = [0.705486, 205, 0.83, 0.54, 0.39, 0.0456];

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
    Max_In = [3.995,0.08, 0.054,640, 0.007, 650;
              3.91, 0.072,0.054,680, 0.008, 750;
              3.82, 0.072,0.058,750, 0.0075,800;
              3.72, 0.075,0.065,700, 0.0085,900;
              3.62, 0.08, 0.07, 750, 0.009, 950;
              3.5,  0.09, 0.07, 700, 0.01,  900;
              3.36, 0.1,  0.07, 500, 0.012, 600;
              3.27, 0.105,0.06, 550, 0.017, 400;
              3.17, 0.11, 0.05, 550, 0.022, 300;
              3.08, 0.115,0.038,600, 0.026, 200;
              2.99, 0.12, 0.03, 600, 0.025, 180;
              2.9,  0.11, 0.03, 400, 0.016, 300;
              2.84, 0.1,  0.014,250, 0.014, 250];

    Min_In = [3.955,0.055,0.044,560, 0.006, 400;
              3.87, 0.056,0.04, 520, 0.005, 500;
              3.76, 0.054,0.04, 500, 0.005, 500;
              3.64, 0.055,0.04, 450, 0.0045,500;
              3.52, 0.055,0.04, 450, 0.004, 450;
              3.38, 0.06, 0.045,400, 0.005, 300;
              3.28, 0.07, 0.052,350, 0.006, 250;
              3.18, 0.075,0.044,300, 0.009, 150;
              3.09, 0.08, 0.035,300, 0.011, 100;
              2.98, 0.08, 0.028,300, 0.013, 100;
              2.91, 0.08, 0.02, 200, 0.015, 100;
              2.86, 0.08, 0.01, 100, 0.008, 100;
              2.79, 0.06, 0.008,150, 0.008, 100];

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
    currentFile = sprintf('CNN_Result_2_60_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end