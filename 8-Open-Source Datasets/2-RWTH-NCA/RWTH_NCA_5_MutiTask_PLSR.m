clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 2-RWTH-NCA
%%% This script: Assemble RC-type relaxation features and expanded health
%%% indicators, normalize inputs/outputs, and run repeated leave-one-out
%%% PLSR to predict multi-dimensional ageing states. Prediction results are
%%% saved as MAT-files for downstream analysis and visualization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('OneCycle_NCA_RWTH.mat')
load('Feature_ALL_RWTH_NCA.mat')

% Assemble feature tensor: (sample, relaxation-segment index, feature-dimension)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build target outputs from cycle-level records (capacity, life, and expanded indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: use the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (cycle-level, consistent with the original script)
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to health-indicator scales
% Keep the same scaling/normalization conventions as used in the original script
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

    % Mapping from CountSV to the feature index used in this script (kept as original)
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

    % Repeated leave-one-out evaluation loop
    for CountRP = 1:100
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

            % 2. Train PLSR (automatic component selection)
            ncomp = min(5, size(In_Train,1));  % at most 5 components
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);

            % 3. Predict
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';

            toc
        end

        % Clip predicted outputs to [0, 1] in the normalized space
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
    currentFile = sprintf('PLSR_Result_1_80_Y_Test_%d_RWTH_NCA.mat',CountSV);
    save(currentFile,'Y_Test')
end