clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: 3RC feature set (Uoc, R0, R1, C1, R2, C2, R3, C3) -> PLSR
%%% multi-task ageing assessment on SCU3 Dataset #1 using repeated leave-one-out.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset and extracted 3RC features
load('../../OneCycle_1.mat')  
load('Feature_1_ALL_3RC.mat')

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
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
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

%% PLSR (per voltage setpoint)
for CountSV = 13:-1:1 
    %% Feature normalization (min-max to [0,1] per setpoint)
    Max_In = [4.09, 0.05, 0.045, 2000, 0.006, 2000, 0.004, 2400;
              4.005,0.045,0.04,  2500, 0.008, 2000, 0.002, 3000;
              3.92, 0.046,0.035, 3500, 0.008, 2300, 0.002, 3500;
              3.81, 0.045,0.035, 3500, 0.008, 2300, 0.002, 3500;
              3.73, 0.05, 0.04,  3500, 0.008, 2800, 0.002, 4000;
              3.63, 0.05, 0.035, 6000, 0.018, 2800, 0.004, 5000;
              3.52, 0.06, 0.035, 7000, 0.018, 3100, 0.003, 4000;
              3.43, 0.07, 0.05,  4300, 0.025, 3500, 0.004, 3000;
              3.25, 0.075,0.045, 1300, 0.001, 1500, 0.005, 2300;
              3.2,  0.08, 0.04,  2000, 0.01,  1200, 0.008, 1400;
              3.18, 0.075,0.03,  2500, 0.012, 1400, 0.009, 900;
              3.1,  0.08, 0.025, 2500, 0.013, 4500, 0.007, 9000;
              3.05, 0.07, 0.02,  4000, 0.012, 4000, 0.008, 18000];

    Min_In = [4.04, 0.03, 0.024, 800,  0.003, 900.  0.0005,800;
              3.95, 0.03, 0.02,  1000, 0.0025,1000, 0,     1000;
              3.86, 0.03, 0.015, 1000, 0.002, 1000, 0.0004,1000;
              3.75, 0.03, 0.015, 1000, 0.002, 1000, 0.0004,1000;
              3.66, 0.03, 0.015, 1000, 0.002, 1200, 0.0004,1200;
              3.54, 0.03, 0,     1000, 0.002, 1200, 0,     0;
              3.43, 0.03, 0,     1000, 0,     1200, 0,     1200;
              3.29, 0.04, 0,     700,  0,     700,  0,     500;
              3.2,  0.045,0.024, 800,  0.004, 400,  0.0005,300;
              3.1,  0.05, 0,     800,  0.004, 300,  0.001, 200;
              3,    0.05, 0,     0,    0,     200,  0,     200;
              2.9,  0,    0,     0,    0,     0,    0,     0;
              2.8,  0,    0,     0,    0,     0,    0,     0];

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
    %     axis([0,100,0,1])
    % end
    
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
    currentFile = sprintf('PLSR_Result_1_70_Y_Test_%d_3RC.mat',CountSV);
    save(currentFile,'Y_Test')
end