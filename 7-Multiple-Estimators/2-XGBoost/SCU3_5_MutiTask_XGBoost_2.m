clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Assemble relaxation-feature tensors and normalized outputs
%%% for SCU3 Dataset #2, then run repeated leave-one-out XGBoost regression
%%% across multiple voltage setpoints and save prediction results.
%%% Note: XGBoost MATLAB wrappers are loaded via the local "XGBoost" folder.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset and pre-extracted relaxation features
load('../../OneCycle_2.mat')  
load('../../Feature_2_ALL.mat')

% Add local XGBoost MATLAB wrapper to the search path
addpath('XGBoost')

%% Feature tensor assembly
% Feature(:,:,k) stores the k-th relaxation-derived parameter across
% samples (row) and voltage setpoints (column).
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Filter samples by the ending step flag and extract life, original capacity,
% and expanded health indicators (used as multi-dimensional outputs).
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
        % Original capacity (Ah)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (cycle-index selection follows raw data structure)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Unified health-indicator scaling
% Convert raw variables into comparable normalized health-indicator forms
% (these are not the final [0,1] normalization used by the learning model).
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization
% Apply min-max normalization to construct the multi-dimensional output matrix
Max_Out = [0.716971, 805, 0.9,  0.83, 0.75, 0.356];
Min_Out = [0.705486, 205, 0.83, 0.54, 0.39, 0.0456];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0,1]
Output(Output<0) = 0;Output(Output>1) = 1;

%% XGBoost
% For each voltage setpoint index (CountSV), normalize input features using
% setpoint-specific bounds, then run repeated leave-one-out evaluation.
for CountSV = 13:-1:1
    %% Input feature normalization (setpoint-dependent)
    % Max_In/Min_In provide per-setpoint bounds for the 6 relaxation features
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

    % Map CountSV (13..1) to row index (1..13) in bound tables
    MyInd = 14-CountSV;
          
    % Min-max normalize the 6 relaxation features at the current setpoint
    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,CountSV,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,CountSV,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized inputs/outputs to [0,1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    % Repeated leave-one-out evaluation (CountRP repetitions)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            % Progress printing (kept as in original script)
            CountSV
            CountRP
            IndexData

            % Construct LOOCV split: one sample for test, remaining for training
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Ensure 2-D shapes are consistent for xgboost_train/xgboost_test
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% XGBoost model configuration
            % Models are trained independently for each output dimension.
            num_trees = 100;
            params.eta = 0.1;
            params.objective = 'reg:logistic';
            params.max_depth = 5;
            params.min_child_weight  = 0.3;
            params.num_parallel_tree = 3;
            params.subsample = 0.3;

            model_1 = xgboost_train(In_Train', Out_Train(1,:)', params, num_trees);
            model_2 = xgboost_train(In_Train', Out_Train(2,:)', params, num_trees);
            model_3 = xgboost_train(In_Train', Out_Train(3,:)', params, num_trees);
            model_4 = xgboost_train(In_Train', Out_Train(4,:)', params, num_trees);
            model_5 = xgboost_train(In_Train', Out_Train(5,:)', params, num_trees);
            model_6 = xgboost_train(In_Train', Out_Train(6,:)', params, num_trees);

            % Prediction (train and held-out test sample) in normalized space
            Y_Train(1,:) = xgboost_test(In_Train', model_1)';
            Y_Test(CountRP,1,IndexData) = xgboost_test(In_Test(:,IndexData)', model_1)';

            Y_Train(2,:) = xgboost_test(In_Train', model_2)';
            Y_Test(CountRP,2,IndexData) = xgboost_test(In_Test(:,IndexData)', model_2)';

            Y_Train(3,:) = xgboost_test(In_Train', model_3)';
            Y_Test(CountRP,3,IndexData) = xgboost_test(In_Test(:,IndexData)', model_3)';

            Y_Train(4,:) = xgboost_test(In_Train', model_4)';
            Y_Test(CountRP,4,IndexData) = xgboost_test(In_Test(:,IndexData)', model_4)';

            Y_Train(5,:) = xgboost_test(In_Train', model_5)';
            Y_Test(CountRP,5,IndexData) = xgboost_test(In_Test(:,IndexData)', model_5)';

            Y_Train(6,:) = xgboost_test(In_Train', model_6)';
            Y_Test(CountRP,6,IndexData) = xgboost_test(In_Test(:,IndexData)', model_6)';
        
            toc
        end

        % Clip predictions to [0,1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% De-normalization
        % Map predictions and labels back to the original output scales
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
        
        %% Result visualization
        % Scatter plots of prediction vs. label for training and test sets
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
        
    end

    % Save repeated-test predictions for the current voltage setpoint
    currentFile = sprintf('XGBoost_Result_2_60_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end