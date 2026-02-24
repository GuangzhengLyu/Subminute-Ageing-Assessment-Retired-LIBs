clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: 4RC feature set (Uoc, R0, R1, C1, R2, C2, R3, C3, R4, C4)
%%% -> PLSR multi-task ageing assessment on SCU3 Dataset #1 using repeated
%%% leave-one-out per voltage setpoint (3.0 V to 4.2 V).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset and pre-extracted 4RC feature matrices
load('../../OneCycle_1.mat')  
load('Feature_1_ALL_4RC.mat')

% Assemble the RC feature tensor: Feature(sample, setpoint, feature_index)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;
Feature(:,:,7) = R3;
Feature(:,:,8) = C3;
Feature(:,:,9) = R4;
Feature(:,:,10) = C4;

%% Sample construction
% Select valid samples and construct multi-task outputs:
% [capacity-based SOH, RUL proxy, energy efficiency, CC charge ratio,
%  mid-point voltage, platform discharge capacity]
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % RUL proxy: first cycle index where discharge capacity drops below the threshold
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH (normalized later)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Unify as health indicators (normalization to [0,1] scale bases)
Capa      = Capa/3.5;
Life      = Life;
ERate     = ERate/89;
CoChRate  = CoChRate/83;
MindVolt  = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization (min-max to [0,1])
Max_Out = [0.99, 450, 1.01, 1.02, 1.04, 1.05];
Min_Out = [0.79, 100, 0.94, 0.9,  0.86, 0.65];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0,1] in normalized space
Output(Output<0) = 0;Output(Output>1) = 1;

%% PLSR (per voltage setpoint)
% CountSV: setpoint index (13 -> 1), corresponding to 3.0 V -> 4.2 V
for CountSV = 13:-1:1 
    %% Feature normalization (min-max to [0,1] per setpoint)
    % Rows correspond to setpoint index MyInd = 14-CountSV (3.0 V -> 4.2 V).
    Max_In = [4.09, 0.05, 0.025, 5000, 0.025, 6000, 0.012, 2200, 0.0055, 2500;
              4.005,0.048,0.03,  5000, 0.025, 8000, 0.01,  2400, 0.004,  4000;
              3.92, 0.048,0.025, 6500, 0.016, 9000, 0.008, 2400, 0.0016, 4500;
              3.815,0.045,0.02,  7000, 0.025, 8000, 0.008, 2500, 0.003,  4500;
              3.73, 0.05, 0.018, 8000, 0.016, 7000, 0.008, 3000, 0.0016, 4000;
              3.63, 0.05, 0.018, 9000, 0.018, 10000,0.012, 3000, 0.0035, 7000;
              3.51, 0.055,0.03,  10000,0.018, 7000, 0.012, 3000. 0.0035, 6000;
              3.45, 0.07, 0.03,  6000, 0.025, 5000, 0.014, 4500, 0.005,  3000;
              3.26, 0.075,0.04,  4500, 0.025, 3500, 0.01,  1600, 0.005,  2500;
              3.2,  0.075,0.03,  6000, 0.02,  3500, 0.009, 1100, 0.007,  1400;
              3.2,  0.075,0.03,  6000, 0.02,  3500, 0.01,  2000, 0.006,  10000;
              3.1,  0.08, 0.025, 5000, 0.018, 5000, 0.012, 10000,0.008,  80000;
              3.05, 0.08, 0.02,  4000, 0.014, 4000, 0.014, 8000, 0.014,  70000];

    Min_In = [4.04, 0.03, 0.01,  1500, 0,     1000. 0,     800,  0.0005, 1000;
              3.94, 0.03, 0.01,  1500, 0,     1000, 0,     900,  0,      800;
              3.86, 0.03, 0.008, 1500, 0,     2000, 0,     1000, 0.0003, 1000;
              3.75, 0.03, 0.008, 2000, 0.005, 1800, 0.002, 1000, 0.0005, 1000;
              3.65, 0.03, 0.006, 2000, 0.005, 1000, 0.001, 1200, 0.0004, 1500;
              3.54, 0.03, 0,     2000, 0,     2000, 0,     1200, 0,      1000;
              3.44, 0.03, 0,     2000, 0,     2000, 0,     1000, 0,      1500;
              3.3,  0.04, 0,     1000, 0,     1000, 0,     800,  0,      1000;
              3.2,  0.045,0.01,  1000, 0.005, 1000, 0.002, 400,  0.0005, 0;
              3.09, 0.05, 0,     0,    0,     300,  0,     300,  0.001,  300;
              3,    0.05, 0,     0,    0,     200,  0,     200,  0,      0;
              2.9,  0,    0,     0,    0,     0,    0,     0,    0,      0;
              2.8,  0,    0,     0,    0,     0,    0,     0,    0,      0];

    MyInd = 14-CountSV;
          
    Feature_Nor(:,1)  = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2)  = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3)  = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4)  = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5)  = (Feature(:,CountSV,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6)  = (Feature(:,CountSV,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    Feature_Nor(:,7)  = (Feature(:,CountSV,7)-Min_In(MyInd,7))/(Max_In(MyInd,7)-Min_In(MyInd,7));
    Feature_Nor(:,8)  = (Feature(:,CountSV,8)-Min_In(MyInd,8))/(Max_In(MyInd,8)-Min_In(MyInd,8));
    Feature_Nor(:,9)  = (Feature(:,CountSV,9)-Min_In(MyInd,9))/(Max_In(MyInd,9)-Min_In(MyInd,9));
    Feature_Nor(:,10) = (Feature(:,CountSV,10)-Min_In(MyInd,10))/(Max_In(MyInd,10)-Min_In(MyInd,10));

    % for j = 1:10
    %     figure(j),clf,plot(Feature_Nor(:,j))
    %     axis([0,100,0,1])
    % end
    
    % Clip normalized features and outputs to [0,1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    %% Repeated leave-one-out PLSR
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Leave-one-out split
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            % Train PLSR (limit components for RC feature dimensionality)
            ncomp = min(3, size(In_Train,1));
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);
            
            % Predict train and test
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';
        
            toc
        end

        % Clip predictions to [0,1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% Inverse-normalize outputs (back to indicator scales)
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
        
%         % Optional visualization (disabled for batch runs)
%         for IndOTrain = 1:6
%             figure(IndOTrain),hold on
%             plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
%             
%             figure(IndOTrain+6),hold on
%             plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
%         end
    end

    % Save predictions for this setpoint index
    currentFile = sprintf('PLSR_Result_1_70_Y_Test_%d_4RC.mat',CountSV);
    save(currentFile,'Y_Test')
end