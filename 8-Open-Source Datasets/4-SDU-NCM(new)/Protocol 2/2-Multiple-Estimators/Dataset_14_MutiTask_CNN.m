clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Build normalized multi-task targets for SDU-NCM (Part P2),
%%% normalize relaxation-parameter inputs, and run repeated leave-one-out
%%% evaluation using a simple CNN-style regressor (trainNetwork).
%%% The per-sample input is reshaped into an "image-like" tensor for the CNN.
%%% Predictions are clamped to [0, 1], denormalized for visualization, and
%%% saved for downstream RMSE aggregation and cross-model comparison.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and extracted relaxation features
load('../OneCycle_SDU_NCM_P2.mat')  
load('../Feature_ALL_SDU_NCM_P2.mat')

% Stack relaxation features into a 3D tensor for consistent downstream access
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

    % Define life as the available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Normalize to unified health-indicator scale
% Note: Scaling factors and ranges are preserved as in the original script
Capa = Capa/2.4;
Life = Life;
ERate = ERate/0.95;
CoChRate = CoChRate/0.9;
MindVolt = (MindVolt-2.65)/(3.65-2.65);
PlatfCapa = PlatfCapa/1.8;

%% Output normalization (targets)
% Output columns correspond to:
% 1) capacity-based SOH proxy, 2) life (cycles), 3) energy rate,
% 4) constant-charge rate, 5) mid-point voltage, 6) platform discharge capacity
Max_Out = [1.01,  550, 0.975, 1.006, 1.026, 0.89];
Min_Out = [0.975, 50,  0.94,  0.992, 1.014, 0.855];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clamp normalized outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Input feature normalization (relaxation-parameter features)
    % Normalization ranges are manually specified and preserved as-is
    Max_In = [4.1922, 0.057,  0.12, 1250, 0.055, 1300];
    Min_In = [4.1908, 0.051, 0.05, 750,  0.01,  900];

    % Index mapping (kept as in original script)
    MyInd = 15-CountSV;
          
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clamp normalized inputs and outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    for CountRP = 1:100
        % Repeated leave-one-out evaluation (repeat CountRP times)
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create leave-one-out split by removing the current sample
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Pack current test sample into an "image-like" input tensor
            % Note: The original script uses size(Temp_F,3); keep as-is
            for NumST = 1:size(Temp_F,3)
                In_Test(NumST,:,1,1) = Temp_F(IndexData,:,NumST);
            end
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Remove test sample (train)
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            % Pack training samples into an "image-like" input tensor
            for NumST = 1:size(Temp_F,3)
                for NumID = 1:size(Temp_F,1)
                    In_Train(NumST,:,1,NumID) = Temp_F(NumID,:,NumST);
                end
            end
            Out_Train = Temp_O';
            
            % Define a compact CNN-style regression network
            layers = [
                imageInputLayer([size(In_Train,1) 6 1])
                % Convolution block
                convolution2dLayer([ceil(size(In_Train,1)/2),3],20,'Padding','same')
                maxPooling2dLayer([1,1],'Stride',2)
                % Fully-connected regression head
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];
            
            % Training configuration (kept as in original script)
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

            % Train model on the current split
            net = trainNetwork(In_Train,Out_Train',layers,options);

            % Predict on training set and the held-out test sample (multi-output)
            Y_Train = predict(net,In_Train)';
            Y_Test(CountRP,:,IndexData) = predict(net,In_Test);
        
            toc
        end

        % Clamp predicted normalized outputs to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        % Denormalize predictions and ground truth for plotting/interpretation
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
        
        % Visualization: training fit and leave-one-out test predictions
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save repeated-test predictions for this CountSV setting
    currentFile = sprintf('Result_14_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end