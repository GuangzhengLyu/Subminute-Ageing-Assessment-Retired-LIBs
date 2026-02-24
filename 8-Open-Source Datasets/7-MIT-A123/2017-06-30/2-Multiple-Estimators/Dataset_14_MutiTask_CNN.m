clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123
%%% This script: Train a CNN-based regressor on the MITA123_2 dataset. The CNN
%%% takes a 2D "feature image" constructed from normalized relaxation features
%%% and predicts six normalized health indicators. The workflow uses repeated
%%% leave-one-out splits, visualizes train/test scatter results, and saves
%%% predicted test outputs to MAT-files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and precomputed relaxation features
load('../OneCycle_MITA123_2.mat')
load('../Feature_ALL_MITA123_2.mat')

% Assemble feature tensor: each slice corresponds to one relaxation feature
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build target outputs (capacity/life + expanded performance indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Trajectory length as life proxy (cycle count)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health indicators (dataset-specific references)
% Note: The scaling constants are preserved as in the original script
Capa = Capa/1.1;
Life = Life;
ERate = ERate/0.9;
CoChRate = CoChRate/0.98;
MindVolt = (MindVolt-2)/(3.12-2);
PlatfCapa = PlatfCapa/0.97;

%% Output normalization to [0, 1]
% Define min/max bounds for each output dimension
Max_Out = [0.99, 700, 0.97, 0.994, 0.98, 0.94];
Min_Out = [0.88, 100, 0.93, 0.982, 0.92, 0.76];

% Capacity-based output
Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));

% Life-based output
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip to [0, 1] to avoid out-of-range values after normalization
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14
    %% Input feature normalization to [0, 1]
    % Define min/max bounds for each input feature dimension
    Max_In = [2.55, 4,   14, 25, 14, 0.7];
    Min_In = [2.35, 0.5, 4,  0,  10, 0.2];

    % Select the row of bounds (structure preserved as in the original script)
    MyInd = 15-CountSV;

    % Normalize each feature dimension (fixed relaxation segment index = 1)
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clip normalized inputs and outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out CNN regression
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create leave-one-out split (test = current sample; train = rest)
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Build CNN test input (feature image)
            for NumST = 1:size(Temp_F,3)
                In_Test(NumST,:,1,1) = Temp_F(IndexData,:,NumST);
            end

            % Store ground truth of the left-out sample
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove the test sample from training pool
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Build CNN training input (feature images)
            for NumST = 1:size(Temp_F,3)
                for NumID = 1:size(Temp_F,1)
                    In_Train(NumST,:,1,NumID) = Temp_F(NumID,:,NumST);
                end
            end

            % Arrange training targets as (outputs x samples)
            Out_Train = Temp_O';

            %% CNN definition (2D "feature image" regression)
            layers = [
                imageInputLayer([size(In_Train,1) 6 1])

                % Convolution block
                convolution2dLayer([ceil(size(In_Train,1)/2),3],20,'Padding','same')
                maxPooling2dLayer([1,1],'Stride',2)

                % Fully-connected regression head
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];

            % Training options (Adam optimizer), preserved as in the original script
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
                'Plots',               'none'...
                );

            % Train network on the leave-one-out training set
            net = trainNetwork(In_Train,Out_Train',layers,options);

            % Predict (training set and the left-out test sample)
            Y_Train = predict(net,In_Train)';
            Y_Test(CountRP,:,IndexData) = predict(net,In_Test);

            toc
        end

        % Clip predicted normalized outputs to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% De-normalize predictions and ground truth back to physical units
        % Capacity (Output 1) and life (Output 2) are handled explicitly
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

        %% Visualization: ground truth vs prediction (train and test)
        % Figures 1–6: training scatter; Figures 7–12: test scatter
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save predicted test outputs for this CountSV setting
    currentFile = sprintf('Result_14_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end