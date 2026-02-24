clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Conventional Full-Charge Relaxation Voltage vs Pulse-Inspection
%%%          Relaxation Voltage
%%% This script: Load SCU3 Dataset #1 and EC-model features (Uoc, R0, R1, C1,
%%% R2, C2), build normalized multi-target outputs (capacity, life, and
%%% expanded indicators), run repeated leave-one-out PLSR, and save Y_Test.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset
load('../OneCycle_1.mat')  
load('Feature_1_EC.mat')

% Assemble feature tensor: [sample, setpoint-index, feature-dimension]
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Filter samples by the ending step flag and extract raw targets:
% capacity, life, and expanded health indicators.
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 2.5 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Remaining capacity proxy (original capacity at the selected cycle)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (raw values)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Normalize targets to unified health indicators
% Fixed reference scaling used for multi-task regression
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization to [0, 1]
% Target-specific bounds for normalization (clipped to [0, 1])
Max_Out = [0.99, 450, 1.01, 1.02, 1.04, 1.05];
Min_Out = [0.79, 100, 0.94, 0.9,  0.86, 0.65];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

Output(Output<0) = 0;Output(Output>1) = 1;

%% PLSR
% This script runs the selected setpoint index only (CountSV = 14).
for CountSV = 14:-1:14 
    %% Input feature normalization to [0, 1]
    % Feature-specific bounds for normalization (single setpoint configuration)
    Max_In = [4.19, 0.0065, 0.035,  45000, 0.005, 45000];
    Min_In = [4.13, 0.0015, 0,      10000, 0,     0];

    MyInd = 15-CountSV;
          
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized features/targets to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    % Repeated leave-one-out evaluation
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Build train/test split by removing one sample each time
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Ensure consistent 2D shapes for PLSR
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            % Train PLSR (component count capped at 5)
            ncomp = min(5, size(In_Train,1));
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);
            
            % Predict on train and test sets
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';
        
            toc
        end

        % Clip predictions to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        % De-normalize predictions and targets back to physical scales
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
        
        % Result visualization: train (1-6) and test (7-12)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save test predictions for the current setpoint index
    currentFile = sprintf('./Result/PLSR_Result_1_70_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end