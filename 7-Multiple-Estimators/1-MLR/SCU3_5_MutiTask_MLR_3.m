clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Assemble relaxation-feature tensors and normalized outputs
%%% for SCU3 Dataset #3, then run repeated leave-one-out MLR (fitlm) across
%%% multiple voltage setpoints with channel-wise exclusion (to mitigate RUL
%%% leakage) and save prediction results for each setpoint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset and pre-extracted relaxation features
load('../../OneCycle_3.mat')  
load('../../Feature_3_ALL.mat')

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

        % Define life as the first cycle where discharge capacity drops below 1.75 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity
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
% (these are not the final [0,1] normalization used by MLR).
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization
% Apply min-max normalization to construct the multi-dimensional output matrix
Max_Out = [0.72, 1800, 0.9,  0.8,  0.75, 0.4];
Min_Out = [0.57, 400,  0.74, 0.15, 0,    0];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0,1]
Output(Output<0) = 0;Output(Output>1) = 1;

%% MLR
% For each voltage setpoint index (CountSV), normalize input features using
% setpoint-specific bounds, then run repeated leave-one-out evaluation.
for CountSV = 13:-1:1  

    %% Input feature normalization (setpoint-dependent)
    % Max_In/Min_In provide per-setpoint bounds for the 6 relaxation features
    Max_In = [4,    0.16, 0.085, 650, 0.05,  700;
              3.92, 0.18, 0.1,   700, 0.04,  750;
              3.83, 0.17, 0.1,   700, 0.05,  800;
              3.73, 0.18, 0.09,  750, 0.05,  900;
              3.63, 0.18, 0.09,  700, 0.05,  900;
              3.5,  0.17, 0.085, 650, 0.055, 850;
              3.37, 0.17, 0.072, 550, 0.055, 600;
              3.28, 0.17, 0.062, 550, 0.05,  350;
              3.18, 0.17, 0.05,  600, 0.05,  300;
              3.08, 0.155,0.04,  600, 0.04,  450;
              2.99, 0.142,0.034, 600, 0.03,  400;
              2.94, 0.13, 0.035, 450, 0.02,  280;
              2.9,  0.11, 0.02,  280, 0.02,  280];
    Min_In = [3.7,  0.05, 0.04,  350, 0.005, 0;
              3.6,  0.04, 0.04,  250, 0.005, 0;
              3.48, 0.05, 0.04,  200, 0.005, 0;
              3.38, 0.05, 0.04,  200, 0.004, 0;
              3.28, 0.05, 0.04,  200, 0.004, 0;
              3.2,  0.06, 0.045, 200, 0.005, 0;
              3.13, 0.07, 0.052, 200, 0.005, 50;
              3.05, 0.07, 0.042, 200, 0.01,  50;
              2.99, 0.08, 0.033, 0,   0.01,  50;
              2.94, 0.08, 0.024, 50,  0.015, 50;
              2.89, 0.08, 0.014, 60,  0.006, 60;
              2.84, 0.04, 0.008, 70,  0.004, 90;
              2.78, 0.02, 0.004, 100, 0.004, 110];

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

            % Construct test sample
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Channel-wise exclusion (avoid training/testing leakage for RUL)
            CountSamChannel = 0;
            for IndDataCn = 1:length(Feature_Nor)
                if OneCycle(IndDataCn).Channel==OneCycle(IndexData).Channel
                    CountSamChannel = CountSamChannel+1;
                    SamChannelIndx(CountSamChannel) = IndDataCn;
                end
            end
            Temp_F(SamChannelIndx,:) = [];
            Temp_O(SamChannelIndx,:) = [];

            % Training matrices
            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Ensure 2-D shapes are consistent for fitlm and predict
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            %% Train MLR models (one model per output dimension)
            MLR_SOH_Capa = fitlm(In_Train',Out_Train(1,:)');
            MLR_RUL      = fitlm(In_Train',Out_Train(2,:)');
            MLR_SOH_EgyR = fitlm(In_Train',Out_Train(3,:)');
            MLR_SOH_CCCR = fitlm(In_Train',Out_Train(4,:)');
            MLR_SOH_MidV = fitlm(In_Train',Out_Train(5,:)');
            MLR_SOH_PDCa = fitlm(In_Train',Out_Train(6,:)');

            %% Predict on held-out test sample
            Y_Test(CountRP,1,IndexData) = predict(MLR_SOH_Capa,In_Test(:,IndexData)');
            Y_Test(CountRP,2,IndexData) = predict(MLR_RUL,     In_Test(:,IndexData)');
            Y_Test(CountRP,3,IndexData) = predict(MLR_SOH_EgyR,In_Test(:,IndexData)');
            Y_Test(CountRP,4,IndexData) = predict(MLR_SOH_CCCR,In_Test(:,IndexData)');
            Y_Test(CountRP,5,IndexData) = predict(MLR_SOH_MidV,In_Test(:,IndexData)');
            Y_Test(CountRP,6,IndexData) = predict(MLR_SOH_PDCa,In_Test(:,IndexData)');

            toc

            % Keep variable clearing behavior as in original script
            clear Y_Train Out_Train
        end

        % Clip predictions to [0,1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        % De-normalize predictions and labels back to the original output scales
        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end

        % Optional result visualization (kept as in original script)
        for IndOTrain = 1:6
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save repeated-test predictions for the current voltage setpoint
    currentFile = sprintf('MLR_Result_3_50_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end