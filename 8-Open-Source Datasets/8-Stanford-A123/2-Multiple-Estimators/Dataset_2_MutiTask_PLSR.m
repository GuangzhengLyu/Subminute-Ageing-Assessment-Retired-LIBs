clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 8-Stanford-A123
%%% This script: Build health-indicator targets and relaxation-parameter
%%% inputs, apply min-max normalization, and run repeated leave-one-out PLSR
%%% (plsregress) to generate predictions saved as MAT-files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_MITA123_4.mat')  
load('../Feature_ALL_MITA123_4.mat')

% Assemble feature tensor (samples × segments × features)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction (ground-truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: full discharge-capacity trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators (kept as in the original script)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health indicators (dataset-specific scaling)
% Note: Keep the original scaling factors to ensure consistent evaluation.
Capa = Capa/1.1;
Life = Life;
ERate = ERate/0.89;
CoChRate = CoChRate/0.99;
MindVolt = (MindVolt-2)/(3.12-2);
PlatfCapa = PlatfCapa/0.97;

%% Output normalization (targets)
% Output columns:
% 1) Capacity, 2) Life, 3) Energy rate, 4) Constant-charge rate,
% 5) Minimum voltage, 6) Platform capacity
Max_Out = [0.975, 1100, 0.995, 0.993, 0.99, 0.97];
Min_Out = [0.94,  400,  0.97,  0.983, 0.96, 0.88];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Extended health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip normalized outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (inputs)
    % Note: This block uses dataset-specific min/max bounds.
    Max_In = [3.5966, -0.034, 0.068, 44, 0.068, 55];
    Min_In = [3.5954, -0.046, 0.052, 30, 0.052, 30];

    % Select which normalization row to use (structure preserved)
    MyInd = 15-CountSV;
          
    % Normalize relaxation features for the selected segment (segment index = 1)
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized inputs/outputs to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    % Repeated experiments (CountRP): repeated leave-one-out evaluation
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Build leave-one-out split for the current test sample
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Remove test sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            % Training data (features × samples, outputs × samples)
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Keep explicit reshape operations (as in original script)
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            % 2. Train PLSR (automatic selection of the number of components)
            ncomp = min(5, size(In_Train,1));  % Up to 5 components
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);
            
            % 3. Predict (training and test)
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';
        
            toc
        end

        % Clip predicted normalized outputs to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% Denormalization (convert back to physical units)
        % Training set
        Y_Train(1,:) = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Train(1,:) = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Y_Train(2,:) = Y_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        Out_Train(2,:) = Out_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        
        % Test set (current repetition)
        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end
        
        %% Visualization
        % Figures 1–6: training scatter (true vs predicted)
        % Figures 7–12: test scatter (true vs predicted)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save prediction results for this CountSV setting
    currentFile = sprintf('Result_2_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end