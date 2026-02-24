clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: KNN baseline on SCU3 Dataset #1 using ECM feature channels.
%%% It constructs six scaled health indicators (targets), normalizes inputs
%%% and outputs, performs leave-one-out evaluation across 13 relaxation-
%%% voltage setpoints, repeats training/prediction 100 times, saves Y_Test.
%%% Notes:
%%% - Each target dimension is trained with an independent KNN regressor.
%%% - Correlation distance is used; NumNeighbors is fixed to 3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
load('../../OneCycle_1.mat')
load('../../Feature_1_ALL.mat')

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

        % Life definition: first cycle where discharge capacity drops below 2.5 Ah
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
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
Max_Out = [0.99, 450, 1.01, 1.02, 1.04, 1.05];
Min_Out = [0.79, 100, 0.94, 0.9,  0.86, 0.65];

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
    Max_In = [4.10, 0.05, 0.04,  1250, 0.005, 1200;
              4.005,0.045,0.035, 1400, 0.005, 1400;
              3.92, 0.044,0.029, 1800, 0.0045,1800;
              3.815,0.045,0.028, 1750, 0.0038,1700;
              3.73, 0.044,0.026, 1800, 0.0036,1900;
              3.715,0.043,0.026, 1800, 0.0036,1800;
              3.62, 0.045,0.03,  2000, 0.0035,2300;
              3.51, 0.05, 0.032, 1900, 0.0035,2500;
              3.39, 0.06, 0.045, 1400, 0.0055,2200;
              3.26, 0.07, 0.036, 1000, 0.007, 1200;
              3.18, 0.075,0.035, 1200, 0.012, 800;
              3.1,  0.075,0.03,  1500, 0.015, 500;
              3.1,  0.08, 0.025, 1500, 0.015, 3500];

    Min_In = [4.04, 0.03, 0.024, 800,  0.003, 700;
              3.955,0.03, 0.02,  900,  0.0025,700;
              3.87, 0.03, 0.014, 1000, 0.002, 800;
              3.765,0.03, 0.014, 1000, 0.002, 600;
              3.66, 0.03, 0.016, 1000, 0.0018,1000;
              3.67, 0.03, 0.016, 1000, 0.0018,1000;
              3.56, 0.03, 0.012, 1000, 0.0015,1100;
              3.45, 0.034,0.017, 900,  0.0014,1200;
              3.3,  0.041,0.024, 600,  0.0017,400;
              3.2,  0.045,0.024, 600,  0.003, 200;
              3.11, 0.05, 0.01,  600,  0.004, 100;
              3,    0.05, 0,     0,    0,     100;
              2.9,  0.055,0,     0,    0,     0];

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
            % (fitcknn supports numeric responses; we treat the outputs as continuous)
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
    currentFile = sprintf('KNN_Result_1_70_Y_Test_%d.mat', CountSV);
    save(currentFile, 'Y_Test')
end