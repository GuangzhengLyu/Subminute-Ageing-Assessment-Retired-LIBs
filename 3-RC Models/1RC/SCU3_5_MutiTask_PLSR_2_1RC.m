clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: 1-RC feature set (Uoc, R0, R1, C1) -> PLSR multi-task ageing
%%% assessment on SCU3 Dataset #2 with leave-one-out CV repeated 100 times.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load('../../OneCycle_2.mat')
load('Feature_2_ALL_1RC.mat')

Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

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
Max_Out = [0.716971, 805, 0.9,  0.83, 0.75, 0.356];
Min_Out = [0.705486, 205, 0.83, 0.54, 0.39, 0.0456];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

Output(Output<0) = 0; Output(Output>1) = 1;

%% PLSR (per voltage setpoint)
for CountSV = 13:-1:1

    %% Feature normalization (min-max to [0,1] per setpoint)
    Max_In = [4.015,0.085,0.048,460;
              3.93, 0.076,0.05, 500;
              3.84, 0.074,0.052,540;
              3.74, 0.08, 0.06, 550;
              3.63, 0.09, 0.065,600;
              3.52, 0.1,  0.07, 500;
              3.4,  0.11, 0.07, 400;
              3.28, 0.12, 0.065,350;
              3.18, 0.12, 0.06, 350;
              3.1,  0.14, 0.06, 300;
              2.99, 0.13, 0.045,280;
              2.9,  0.12, 0.04, 200;
              2.85, 0.11, 0.028,125];

    Min_In = [3.975,0.06, 0.041,410;
              3.885,0.056,0.036,360;
              3.78, 0.056,0.036,360;
              3.65, 0.055,0.035,300;
              3.54, 0.055,0.025,300;
              3.4,  0.06, 0.04, 250;
              3.3,  0.07, 0.048,220;
              3.2,  0.08, 0.04, 200;
              3.09, 0.08, 0.037,150;
              3,    0.08, 0.03, 120;
              2.91, 0.08, 0.025,100;
              2.86, 0.08, 0.02, 80;
              2.79, 0.06, 0.01, 70];

    MyInd = 14-CountSV;

    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,CountSV,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,CountSV,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));

    Feature_Nor(Feature_Nor<0) = 0; Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0; Output(Output>1) = 1;

    %% Repeated leave-one-out PLSR
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            Temp_F = Feature_Nor;
            Temp_O = Output;

            In_Test(:,IndexData)          = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            In_Train  = Temp_F';
            Out_Train = Temp_O';

            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData) = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            % Train PLSR (limit components for 1-RC input dimensionality)
            ncomp = min(3, size(In_Train,1));
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);

            % Predict
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';

            toc
        end

        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0; Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% Inverse-normalize outputs (back to indicator scales)
        Y_Train(1,:)     = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Train(1,:)   = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Y_Train(2,:)     = Y_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        Out_Train(2,:)   = Out_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);

        Y_Test(CountRP,1,:)   = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:)   = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end
    end

    currentFile = sprintf('PLSR_Result_2_60_Y_Test_%d_1RC.mat',CountSV);
    save(currentFile,'Y_Test')
end