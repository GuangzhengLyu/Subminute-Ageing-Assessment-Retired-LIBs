clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: 3RC feature set (Uoc, R0, R1, C1, R2, C2, R3, C3) -> PLSR
%%% multi-task ageing assessment on SCU3 Dataset #3 using repeated leave-one-out
%%% with channel-aware training for RUL-related evaluation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset and extracted 3RC features
load('../../OneCycle_3.mat')  
load('Feature_3_ALL_3RC.mat')

% Assemble the RC feature tensor: Feature(sample, setpoint, feature_index)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;  
Feature(:,:,7) = R3;
Feature(:,:,8) = C3;  

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;
        
        % Expanded health indicators
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Unify as health indicators (normalization to [0,1] scale bases)
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization (min-max to [0,1])
Max_Out = [0.72, 1800, 0.9,  0.8,  0.75, 0.4];
Min_Out = [0.57, 400,  0.74, 0.15, 0,    0];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

Output(Output<0) = 0;Output(Output>1) = 1;

%% PLSR (per voltage setpoint)
for CountSV = 13:-1:1  
    %% Feature normalization (min-max to [0,1] per setpoint)
    Max_In = [4,    0.16, 0.1,   900, 0.035, 900,   0.025, 1200;
              3.92, 0.17, 0.11,  1000,0.03,  1000,  0.025, 1600;
              3.83, 0.17, 0.11,  1000,0.03,  1000,  0.03,  1600;
              3.73, 0.18, 0.1,   1000,0.03,  1000,  0.03,  1600;
              3.63, 0.18, 0.09,  1000,0.055, 1200,  0.035, 1600;
              3.5,  0.17, 0.085, 1400,0.04,  1000,  0.03,  1400;
              3.36, 0.17, 0.07,  600, 0.04,  700,   0.03,  900;
              3.26, 0.17, 0.06,  700, 0.04,  500,   0.025, 600;
              3.18, 0.17, 0.042, 800, 0.04,  500,   0.025, 400;
              3.08, 0.155,0.035, 900, 0.03,  650,   0.02,  300;
              2.99, 0.142,0.034, 900, 0.03,  700,   0.03,  1100;
              2.94, 0.13, 0.035, 500, 0.018, 400,   0.016, 800;
              2.9,  0.11, 0.014, 420, 0.014, 450,   0.014, 450];

    Min_In = [3.7,  0.05, 0.04,  400, 0.007, 0,     0,     100;
              3.6,  0.05, 0.04,  300, 0.006, 0,     0,     100;
              3.45, 0.05, 0.03,  250, 0.005, 0,     0,     100;
              3.35, 0.05, 0.03,  250, 0.005, 0,     0,     0;
              3.28, 0.05, 0.03,  250, 0.005, 0,     0,     0;
              3.2,  0.06, 0.05,  250, 0.005, 100,   0,     0;
              3.11, 0.07, 0.05,  300, 0.007, 100,   0,     50;
              3.05, 0.07, 0.04,  300, 0.01,  100,   0.002, 50;
              2.99, 0.08, 0.028, 0,   0.01,  100,   0.004, 60;
              2.94, 0.08, 0.02,  80,  0.01,  100,   0.005, 90;
              2.89, 0.08, 0.01,  80,  0.003, 90,    0,     60;
              2.84, 0.04, 0.005, 90,  0.002, 100,   0.002, 100;
              2.78, 0.02, 0.002, 150, 0.002, 150,   0.002, 150];

    MyInd = 14-CountSV;
          
    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,CountSV,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,CountSV,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    Feature_Nor(:,7) = (Feature(:,CountSV,7)-Min_In(MyInd,7))/(Max_In(MyInd,7)-Min_In(MyInd,7));
    Feature_Nor(:,8) = (Feature(:,CountSV,8)-Min_In(MyInd,8))/(Max_In(MyInd,8)-Min_In(MyInd,8));
    
    % for j = 1:8
    %     figure(j),clf,plot(Feature_Nor(:,j))
    %     axis([0,433,0,1])
    % end

    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out PLSR (channel-aware training split)
    for CountRP = 1:100
        for IndexData = 1 :length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
            
           % For RUL-related evaluation, train only on samples from other channels
           CountSamChannel = 0;
            for IndDataCn = 1:length(Feature_Nor)
                if OneCycle(IndDataCn).Channel==OneCycle(IndexData).Channel
                    CountSamChannel = CountSamChannel+1;
                    SamChannelIndx(CountSamChannel) = IndDataCn;
                end
            end
            Temp_F(SamChannelIndx,:) = [];
            Temp_O(SamChannelIndx,:) = [];
        
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
        
        % Inverse normalization
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
        
%         % Result visualization (optional)
%         for IndOTrain = 1:6
%             figure(IndOTrain),hold on
%             plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
%             
%             figure(IndOTrain+6),hold on
%             plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
%         end
        
    end

    % Save predictions for this setpoint index
    currentFile = sprintf('PLSR_Result_3_50_Y_Test_%d_3RC.mat',CountSV);
    save(currentFile,'Y_Test')
end