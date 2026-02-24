clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 3-SDU-NCM
%%% This script: Load the SDU-NCM single-cycle dataset and precomputed
%%% relaxation features (Uoc, R0, R1, C1, R2, C2). Construct normalized
%%% health indicators (capacity, life length, and expanded performance
%%% indicators), normalize inputs/outputs into [0,1], and run repeated
%%% leave-one-out multi-task regression using Random Forests (TreeBagger).
%%% One TreeBagger regression ensemble is trained per output dimension.
%%% Prediction results (Y_Test) are saved to MAT-files for downstream analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and precomputed relaxation features
load('../OneCycle_SDU_NCM.mat')  
load('../Feature_ALL_SDU_NCM.mat')

% Assemble feature tensor (sample × relaxation-segment × feature-dimension)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction (ground truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life: here defined as the available trajectory length (kept as original)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to normalized health indicators
% Normalize each indicator to a reference scale (kept as original)
Capa     = Capa/2.4;
Life     = Life;
ERate    = ERate/0.95;
CoChRate = CoChRate/0.9;
MindVolt = (MindVolt-2.65)/(3.65-2.65);
PlatfCapa= PlatfCapa/1.8;

%% Output normalization (targets)
% Define per-output normalization bounds for mapping into [0, 1]
Max_Out = [0.77, 2000, 0.96,  0.95, 0.97, 0.9];
Min_Out = [0.68, 200,  0.925, 0.88, 0.92, 0.55];

% Build normalized target matrix: Output(:,k) ∈ [0,1]
Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clamp to [0, 1] to avoid out-of-range values after normalization
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Input normalization (features)
    % Note: the following bounds are manually configured and kept as original.
    % Max_In = [4.1875, -0.09, 0.28, 410, 0.26, 6.5];
    % Min_In = [4.1825, -0.19, 0.18, 300, 0.16, 3];

    Max_In = [4.187, 0.09, 0.3,  950, 0.14, 950];
    Min_In = [4.179, 0.07, 0.08, 450, 0,    600];

    % Select which bound-row to use (kept as original indexing rule)
    MyInd = 15-CountSV;

    % Normalize each feature dimension to [0,1] using the selected bounds
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clamp normalized inputs/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out evaluation (Monte Carlo repeats)
    % CountRP: repeat index; IndexData: left-out sample index
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Copy full dataset (kept as original; used to remove one sample)
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Prepare test sample (one-sample test set)
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove the test sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Training matrices: features (dim × Ntrain), outputs (dim × Ntrain)
            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Keep explicit reshape statements as in original script
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            %% Random Forest (TreeBagger) settings
            trees = 100;                                      % number of trees
            leaf  = 5;                                        % minimum leaf size
            OOBPrediction = 'on';                             % enable out-of-bag error estimation
            OOBPredictorImportance = 'on';                    % compute predictor importance

            Method = 'regression';                            % regression mode (not classification)

            % Train one RF regressor per output dimension
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

            % Predict (train and test)
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

        % Clamp predicted normalized outputs to [0,1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% De-normalization (map back to original indicator scales)
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

        %% Visualization (ground truth vs prediction)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save prediction tensor for this CountSV setting
    currentFile = sprintf('Result_6_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end