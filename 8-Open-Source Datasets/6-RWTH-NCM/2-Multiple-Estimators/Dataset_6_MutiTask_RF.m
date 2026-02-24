clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Build multi-task targets (capacity, life, and expanded
%%% performance indicators), normalize inputs/outputs, and run repeated
%%% leave-one-out Random Forest regression (RF via TreeBagger) using RC
%%% relaxation features as inputs. One RF model is trained per target.
%%% Predictions are de-normalized for visualization and saved to MAT files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_NCM_RWTH.mat')  
load('../Feature_ALL_RWTH_NCM.mat')

% Assemble feature tensor: each slice is one relaxation feature matrix
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build per-sample ground-truth targets from the cycling trajectories
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health-indicator targets
% Apply dataset-specific scaling to map each indicator into a comparable range
Capa = Capa/2;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.9-3);
PlatfCapa = PlatfCapa/1.5;

%% Target normalization (min-max)
% NOTE: Max_Out/Min_Out are fixed normalization bounds used for all samples
Max_Out = [0.925, 1800, 1.012, 0.46, 0.946, 0.344];
Min_Out = [0.89,  1200, 1.002, 0.36, 0.934, 0.326];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clamp normalized targets to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (min-max)
    % NOTE: Max_In/Min_In are fixed normalization bounds for the input features.
    % The indexing pattern (MyInd = 15-CountSV) is preserved from the original script.
    Max_In = [4.095, 2.1,  0.4,  1600, 0.17, 900];
    Min_In = [4.082, 1.96, 0.04, 0,    0.03, 0];

    MyInd = 15-CountSV;

    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clamp normalized features/targets to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out Random Forest regression (RF)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create per-iteration train/test split (leave-one-out)
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove test sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Training matrices (features: rows, samples: columns)
            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Preserve original reshape operations
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            %% RF
            trees = 100;                                      % Number of trees
            leaf  = 5;                                        % Minimum leaf size
            OOBPrediction = 'on';                             % Enable out-of-bag prediction
            OOBPredictorImportance = 'on';                    % Compute predictor importance

            Method = 'regression';                            % Regression mode
            net_1 = TreeBagger(trees, In_Train', Out_Train(1,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_2 = TreeBagger(trees, In_Train', Out_Train(2,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_3 = TreeBagger(trees, In_Train', Out_Train(3,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_4 = TreeBagger(trees, In_Train', Out_Train(4,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_5 = TreeBagger(trees, In_Train', Out_Train(5,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_6 = TreeBagger(trees, In_Train', Out_Train(6,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);

            % Predict
            Y_Train(1,:) = predict(net_1,In_Train');
            Y_Test(CountRP,1,IndexData) = predict(net_1,In_Test(:,IndexData)');
            Y_Train(2,:) = predict(net_2,In_Train');
            Y_Test(CountRP,2,IndexData) = predict(net_2,In_Test(:,IndexData)');
            Y_Train(3,:) = predict(net_3,In_Train');
            Y_Test(CountRP,3,IndexData) = predict(net_3,In_Test(:,IndexData)');
            Y_Train(4,:) = predict(net_4,In_Train');
            Y_Test(CountRP,4,IndexData) = predict(net_4,In_Test(:,IndexData)');
            Y_Train(5,:) = predict(net_5,In_Train');
            Y_Test(CountRP,5,IndexData) = predict(net_5,In_Test(:,IndexData)');
            Y_Train(6,:) = predict(net_6,In_Train');
            Y_Test(CountRP,6,IndexData) = predict(net_6,In_Test(:,IndexData)');

            toc
        end

        % Clamp normalized predictions to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% De-normalization (inverse min-max)
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

        %% Visualization (train vs test scatter)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save predictions for this CountSV setting
    currentFile = sprintf('Result_6_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end
