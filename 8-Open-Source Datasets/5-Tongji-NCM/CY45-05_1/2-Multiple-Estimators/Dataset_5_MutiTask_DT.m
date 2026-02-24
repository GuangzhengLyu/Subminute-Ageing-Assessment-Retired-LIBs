clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 5-Tongji-NCM
%%% This script: Build normalized inputs (relaxation features) and outputs
%%% (capacity/life and expanded performance indicators), then run repeated
%%% leave-one-out Decision Tree Regression (six independent trees) to predict
%%% all health indicators and save the test predictions to a MAT-file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_TongjiNCM.mat')  
load('../Feature_ALL_TongjiNCM.mat')

% Pack relaxation features into a 3D array for consistent downstream handling
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Assemble ground-truth targets (life, capacity, and expanded indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life defined as the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to health-indicator scale (reference normalization)
% Keep the original operations; only clarify intent via comments
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.93;
CoChRate = CoChRate/0.86;
MindVolt = (MindVolt-2.65)/(3.50-2.65);
PlatfCapa = PlatfCapa/1.31;

%% Output normalization (min-max)
% Define output ranges used for scaling each target into [0,1]
Max_Out = [0.946, 1000, 1.005, 1.002, 1.02, 1.01];
Min_Out = [0.926, 200,  0.98,  0.984, 1,    0.975];

% Capacity and life
Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clamp normalized outputs to [0,1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Input feature normalization (min-max)
    % Define input ranges used to scale relaxation features into [0,1]
    Max_In = [4.174,  0.059, 0.043,  125, 0.043, 125];
    Min_In = [4.1705, 0.053, 0.0385, 110, 0.039, 110];

    % Index mapping used by the original script (kept as-is)
    MyInd = 15-CountSV;
          
    % Normalize each relaxation feature using the selected min/max set
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clamp normalized inputs/outputs to [0,1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    %% Repeated leave-one-out Decision Tree Regression
    % Six independent trees are trained (one per target) for each split
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Copy data to temporary matrices for leave-one-out splitting
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Held-out test sample
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Training set (remove the held-out sample)
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            % Arrange as (features x samples) and (targets x samples)
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Preserve original reshape operations (kept as-is)
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            % 2. Train Decision Trees (one tree per output)
            trees = cell(6, 1);
            
            for i = 1:6
                trees{i} = fitrtree(In_Train', Out_Train(i,:)');
                Y_Train(i,:) = predict(trees{i},In_Train')';
                Y_Test(CountRP,i,IndexData) = predict(trees{i},In_Test(:,IndexData)');
            end
        
            toc
        end

        % Clamp predictions to [0,1] before inverse normalization
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        % Inverse normalization (map predictions back to physical units)
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
        
        % Result visualization
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save results for this CountSV setting
    currentFile = sprintf('Result_5_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end