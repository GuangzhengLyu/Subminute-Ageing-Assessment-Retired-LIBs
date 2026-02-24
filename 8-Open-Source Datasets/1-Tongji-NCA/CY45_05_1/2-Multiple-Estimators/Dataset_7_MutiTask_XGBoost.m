clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 1-Tongji-NCA
%%% This script: Assemble relaxation features (Uoc, R0, R1, C1, R2, C2) and
%%% expanded health indicators, apply min–max normalization, and run a
%%% repeated leave-one-out multi-task regression workflow using XGBoost
%%% (MATLAB wrapper functions xgboost_train/xgboost_test). One XGBoost model
%%% is trained per output. The test predictions (Y_Test) are saved as
%%% MAT-files for downstream evaluation and visualization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_TongjiNCA.mat')  
load('../Feature_ALL_TongjiNCA.mat')

% XGBoost library path
addpath('./7-XGBoost');

% Pack relaxation features into a single 3D feature tensor for convenience
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction (ground-truth labels)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full available discharge-capacity trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (as stored in OneCycle)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert raw measurements into normalized health indicators
% NOTE: The scaling factors below are kept exactly as in the original script
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.93;
CoChRate = CoChRate/0.86;
MindVolt = (MindVolt-2.65)/(3.54-2.65);
PlatfCapa = PlatfCapa/1.35;

%% Output normalization (min–max, per indicator)
Max_Out = [0.952, 800, 0.985, 0.994, 0.992, 0.986];
Min_Out = [0.943, 200, 0.976, 0.982, 0.978, 0.966];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip normalized outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (min–max, per relaxation feature)
    % NOTE: The bounds below are kept exactly as in the original script
    Max_In = [4.1718, 0.054, 0.049,  102, 0.049, 102];
    Min_In = [4.1698, 0.046, 0.046,  90,  0.046, 91];

    % Mapping index for per-setting bounds (structure preserved as in original)
    MyInd = 15-CountSV;
          
    % Normalize each relaxation feature to [0, 1]
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized features/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    for CountRP = 1:2
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Leave-one-out split: hold out IndexData as test sample
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Training set: remove the test sample
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Ensure 2D matrix shapes expected by xgboost_train/xgboost_test
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% XGBoost (one model per output)
            num_trees = 100;
            params.eta = 0.1;
            params.objective = 'reg:logistic';
            params.max_depth = 5;
            params.min_child_weight  = 0.3;
            params.num_parallel_tree = 3;
            params.subsample = 0.3;

            % Train XGBoost models (one regressor per output)
            model_1 = xgboost_train(In_Train', Out_Train(1,:)', params, num_trees);
            model_2 = xgboost_train(In_Train', Out_Train(2,:)', params, num_trees);
            model_3 = xgboost_train(In_Train', Out_Train(3,:)', params, num_trees);
            model_4 = xgboost_train(In_Train', Out_Train(4,:)', params, num_trees);
            model_5 = xgboost_train(In_Train', Out_Train(5,:)', params, num_trees);
            model_6 = xgboost_train(In_Train', Out_Train(6,:)', params, num_trees);

            % Predict on training set and held-out test sample
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

        % Clip test predictions to [0, 1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% Denormalize predictions and labels back to original output scales
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
        
        %% Visualization (scatter: ground truth vs prediction)
        % Figures 1–6: training scatter; Figures 7–12: test scatter
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save test predictions for this setting value
    currentFile = sprintf('Result_7_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end