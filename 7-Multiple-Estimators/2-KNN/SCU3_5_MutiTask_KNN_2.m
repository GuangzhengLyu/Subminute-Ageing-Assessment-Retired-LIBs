clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: KNN baseline on SCU3 Dataset #2 using ECM feature channels.
%%% It constructs six scaled health indicators (targets), normalizes inputs
%%% and outputs, performs leave-one-out evaluation across 13 relaxation-
%%% voltage setpoints, repeats training/prediction 100 times, saves Y_Test.
%%% Notes:
%%% - Each target dimension is trained with an independent KNN regressor.
%%% - Correlation distance is used; NumNeighbors is fixed to 3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
load('../../OneCycle_2.mat')
load('../../Feature_2_ALL.mat')

% ECM feature tensor: (sample, voltage_setpoint, feature_dim)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction (ground truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Life definition: first cycle where discharge capacity drops below 2.1 Ah
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Raw capacity and expanded indicators from the first cycle snapshot used
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (raw values)
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Scale targets to unified health indicators
Capa      = Capa/3.5;
Life      = Life;
ERate     = ERate/89;
CoChRate  = CoChRate/83;
MindVolt  = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization to [0,1] using pre-defined bounds
Max_Out = [0.716971, 805, 0.9,  0.83, 0.75, 0.356];
Min_Out = [0.705486, 205, 0.83, 0.54, 0.39, 0.0456];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip to [0,1] to avoid extrapolation issues
Output(Output<0) = 0; Output(Output>1) = 1;

%% KNN: leave-one-out across samples, repeated 100 times, for 13 setpoints
for CountSV = 13:-1:1
    %% Input feature normalization for each voltage setpoint (pre-defined bounds)
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

    MyInd = 14-CountSV;

    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,CountSV,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,CountSV,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clip normalized features to [0,1]
    Feature_Nor(Feature_Nor<0) = 0; Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0; Output(Output>1) = 1;

    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Leave-one-out split
            Temp_F = Feature_Nor;
            Temp_O = Output;

            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            In_Train  = Temp_F';
            Out_Train = Temp_O';

            % Ensure 2-D shapes
            In_Train = reshape(In_Train,[size(In_Train,1), size(In_Train,2)]);
            In_Test(:,IndexData) = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1), size(In_Test(:,IndexData),2)]);

            % Train independent KNN regressors for each output dimension
            knn_1 = fitcknn(In_Train', Out_Train(1,:)', 'NumNeighbors', 3, 'Distance', 'correlation');
            knn_2 = fitcknn(In_Train', Out_Train(2,:)', 'NumNeighbors', 3, 'Distance', 'correlation');
            knn_3 = fitcknn(In_Train', Out_Train(3,:)', 'NumNeighbors', 3, 'Distance', 'correlation');
            knn_4 = fitcknn(In_Train', Out_Train(4,:)', 'NumNeighbors', 3, 'Distance', 'correlation');
            knn_5 = fitcknn(In_Train', Out_Train(5,:)', 'NumNeighbors', 3, 'Distance', 'correlation');
            knn_6 = fitcknn(In_Train', Out_Train(6,:)', 'NumNeighbors', 3, 'Distance', 'correlation');

            % Predictions on train/test (for optional diagnostic plots)
            Y_Train(1,:) = predict(knn_1, In_Train');
            Y_Test(CountRP,1,IndexData) = predict(knn_1, In_Test(:,IndexData)');

            Y_Train(2,:) = predict(knn_2, In_Train');
            Y_Test(CountRP,2,IndexData) = predict(knn_2, In_Test(:,IndexData)');

            Y_Train(3,:) = predict(knn_3, In_Train');
            Y_Test(CountRP,3,IndexData) = predict(knn_3, In_Test(:,IndexData)');

            Y_Train(4,:) = predict(knn_4, In_Train');
            Y_Test(CountRP,4,IndexData) = predict(knn_4, In_Test(:,IndexData)');

            Y_Train(5,:) = predict(knn_5, In_Train');
            Y_Test(CountRP,5,IndexData) = predict(knn_5, In_Test(:,IndexData)');

            Y_Train(6,:) = predict(knn_6, In_Train');
            Y_Test(CountRP,6,IndexData) = predict(knn_6, In_Test(:,IndexData)');

            toc
        end

        % Clip predictions to [0,1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0; Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        % Denormalize train/test outputs back to physical scales
        Y_Train(1,:)    = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Train(1,:)  = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));

        Y_Train(2,:)    = Y_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        Out_Train(2,:)  = Out_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);

        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));

        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end

        % Visualization (optional diagnostics; can be disabled to reduce overhead)
        for IndOTrain = 1:6
            figure(IndOTrain), hold on
            plot(Out_Train(IndOTrain,:), Y_Train(IndOTrain,:), 'o')

            figure(IndOTrain+6), hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)), squeeze(Y_Test(CountRP,IndOTrain,:)), 'o')
        end
    end

    % Save all repeated test predictions for this voltage setpoint
    currentFile = sprintf('KNN_Result_2_60_Y_Test_%d.mat', CountSV);
    save(currentFile, 'Y_Test')
end