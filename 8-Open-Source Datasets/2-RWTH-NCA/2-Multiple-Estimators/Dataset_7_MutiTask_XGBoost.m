clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 2-RWTH-NCA
%%% This script: Assemble RC-type relaxation features and expanded health
%%% indicators, normalize inputs/outputs, and run repeated leave-one-out
%%% multi-task XGBoost regression (one booster per output). Predictions are
%%% clipped, de-normalized, visualized, and saved to MAT-files for downstream
%%% benchmarking and comparison.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_NCA_RWTH.mat')
load('../Feature_ALL_RWTH_NCA.mat')

% XGBoost library functions
addpath('./7-XGBoost');

% Assemble feature tensor: (sample, relaxation-segment index, feature-dimension)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build ground-truth outputs from cycle-level records (capacity, life, and expanded indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: use the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to health-indicator scales
% Keep the same scaling/normalization conventions as used in the original scripts
Capa = Capa/13000;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.7-3);
PlatfCapa = PlatfCapa/5000;

%% Output normalization
% Normalize each output dimension to [0, 1] using fixed min/max bounds
Max_Out = [0.97, 800, 0.94, 0.938, 1,   0.85];
Min_Out = [0.89, 100, 0.87, 0.926, 0.6, 0.8];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14
  %% Input feature normalization
    % Normalize each feature dimension to [0, 1] using fixed min/max bounds
    Max_In = [4.18, 0.105, 200, 35000, 0.2,  10000];
    Min_In = [4.17, 0.06,  0,   5000,  0.02, 2000];

    MyInd = 15-CountSV;

    % Note: The original script indexes Min_In/MyInd as (MyInd,feature),
    % even though Min_In/Max_In are defined as 1-by-6 vectors. This indexing
    % is preserved to avoid changing executable logic.
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clip inputs/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out evaluation loop
    for CountRP = 1:2
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create leave-one-out split (test = current sample; train = remaining)
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Ensure explicit 2D shapes (kept as original operations)
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

          %% XGBoost regression
            num_trees = 100;
            params.eta = 0.1;
            params.objective = 'reg:logistic';
            params.max_depth = 5;
            params.min_child_weight  = 0.3;
            params.num_parallel_tree = 3;
            params.subsample = 0.3;

            % Train one booster per output (kept as explicit models in the original script)
            model_1 = xgboost_train(In_Train', Out_Train(1,:)', params, num_trees);
            model_2 = xgboost_train(In_Train', Out_Train(2,:)', params, num_trees);
            model_3 = xgboost_train(In_Train', Out_Train(3,:)', params, num_trees);
            model_4 = xgboost_train(In_Train', Out_Train(4,:)', params, num_trees);
            model_5 = xgboost_train(In_Train', Out_Train(5,:)', params, num_trees);
            model_6 = xgboost_train(In_Train', Out_Train(6,:)', params, num_trees);

            % Predict on training set (in-sample) and on the held-out sample
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

        % Clip predictions to [0, 1] in the normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        % De-normalization back to the original output scales
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

        % Visualization: train (in-sample) and test (leave-one-out) scatter plots
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save repeated prediction results for this CountSV setting
    currentFile = sprintf('Result_7_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end