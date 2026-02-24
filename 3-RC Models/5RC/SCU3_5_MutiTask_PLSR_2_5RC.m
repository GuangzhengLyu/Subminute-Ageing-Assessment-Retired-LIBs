clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Relaxation-Voltage Feature Extraction
%%% This script: Train and evaluate multi-output PLSR models for SCU3
%%% Dataset #2 using pre-extracted 5-RC relaxation features (Uoc, R0, R1–R5,
%%% C1–C5) at multiple voltage setpoints (13 levels, 3.0–4.2 V). The targets
%%% include normalized capacity SOH, life index (threshold-based), and four
%%% expanded health indicators. A leave-one-out workflow is repeated 100
%%% times per setpoint, and predictions are saved for downstream RMSE plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset and extracted 5-RC feature files
load('../../OneCycle_2.mat')
load('Feature_2_ALL_5RC.mat')

% Pack per-setpoint relaxation features into a tensor:
%   Feature(sample_index, setpoint_index, feature_index)
Feature(:,:,1)  = Uoc;
Feature(:,:,2)  = R0;
Feature(:,:,3)  = R1;
Feature(:,:,4)  = C1;
Feature(:,:,5)  = R2;
Feature(:,:,6)  = C2;
Feature(:,:,7)  = R3;
Feature(:,:,8)  = C3;
Feature(:,:,9)  = R4;
Feature(:,:,10) = C4;
Feature(:,:,11) = R5;
Feature(:,:,12) = C5;

%% Sample construction
% Select valid samples with complete step structure (Steps(end) == 30)
% Define "Life" as the first cycle where discharge capacity drops below 2.1 Ah
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Life definition (threshold-based, dataset #2 uses 2.1 Ah)
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Initial discharge capacity proxy (used as capacity SOH baseline)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (raw scale here; normalized below)
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Convert to normalized health indicators (engineering normalization)
Capa      = Capa/3.5;
Life      = Life;
ERate     = ERate/89;
CoChRate  = CoChRate/83;
MindVolt  = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization (dataset-specific bounds) and clipping to [0,1]
Max_Out = [0.716971, 805, 0.9,  0.83, 0.75, 0.356];
Min_Out = [0.705486, 205, 0.83, 0.54, 0.39, 0.0456];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

Output(Output<0) = 0;Output(Output>1) = 1;

%% PLSR (per voltage setpoint: 13 → 1)
% For each setpoint, normalize features then run repeated leave-one-out PLSR
for CountSV = 13:-1:1

    %% Feature normalization bounds for Uoc and R0 (setpoint-dependent)
    Max_In = [3.985,0.078;
              3.91, 0.074;
              3.81, 0.072;
              3.72, 0.08;
              3.62, 0.085;
              3.5,  0.095;
              3.35, 0.1;
              3.26, 0.11;
              3.17, 0.11;
              3.08, 0.12;
              2.99, 0.12;
              2.9,  0.11;
              2.85, 0.11];

    Min_In = [3.935,0.058;
              3.85, 0.054;
              3.74, 0.054;
              3.62, 0.055;
              3.5,  0.055;
              3.37, 0.06;
              3.25, 0.07;
              3.16, 0.075;
              3.06, 0.08;
              2.98, 0.08;
              2.91, 0.08;
              2.86, 0.08;
              2.79, 0.06];

    % Map setpoint index (13→1) to row index (1→13)
    MyInd = 14-CountSV;

    % Normalize Uoc and R0 by predefined bounds
    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));

    % Robust normalization for remaining RC parameters (R1–R5, C1–C5)
    % Uses sorted positive values to suppress outlier influence
    for i = 3:12
        F_Temp = Feature(:,CountSV,i);
        F_Sort = sort(F_Temp(find(F_Temp>0)));
        Feature_Nor(:,i) = (F_Temp-F_Sort(5))/(F_Sort(end-5)-F_Sort(5));
    end

    % for j = 1:12
    %     figure(j),clf,plot(Feature_Nor(:,j))
    %     axis([0,46,0,1])
    % end

    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out evaluation (100 repeats)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Build leave-one-out split
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            In_Train = Temp_F';
            Out_Train = Temp_O';

            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            % Train PLSR (fixed to up to 3 latent components)
            ncomp = min(3, size(In_Train,1));
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);

            % Predict train and test outputs
            Y_Train = [ones(size(In_Train,2),1), In_Train'] * beta;
            Y_Train = Y_Train';
            Y_Test(CountRP,:,IndexData) = [ones(size(In_Test(:,IndexData),2),1), In_Test(:,IndexData)'] * beta;
            Y_Test(CountRP,:,IndexData) = Y_Test(CountRP,:,IndexData)';

            toc
        end

        % Clip predictions in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        % Denormalize outputs back to original indicator scales
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

%         % Optional parity plots (kept commented as in original style)
%         for IndOTrain = 1:6
%             figure(IndOTrain),hold on
%             plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
%
%             figure(IndOTrain+6),hold on
%             plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
%         end

    end

    % Save per-setpoint prediction tensor for downstream evaluation scripts
    currentFile = sprintf('PLSR_Result_2_60_Y_Test_%d_5RC.mat',CountSV);
    save(currentFile,'Y_Test')
end