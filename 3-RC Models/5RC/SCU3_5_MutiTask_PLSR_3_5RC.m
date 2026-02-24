clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Relaxation-Voltage Feature Extraction
%%% This script: Train and evaluate multi-output PLSR models for SCU3
%%% Dataset #3 using pre-extracted 5-RC relaxation features (Uoc, R0, R1–R5,
%%% C1–C5) at multiple voltage setpoints (13 levels, 3.0–4.2 V). The targets
%%% include normalized capacity SOH, life index (threshold-based), and four
%%% expanded health indicators. A leave-one-out workflow is repeated 100
%%% times per setpoint. For Dataset #3, the training set excludes samples
%%% from the same tester channel as the held-out sample to avoid leakage in
%%% life/RUL-related prediction. Predictions are saved for downstream RMSE
%%% aggregation and sensitivity studies.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset and extracted 5-RC feature files
load('../../OneCycle_3.mat')
load('Feature_3_ALL_5RC.mat')

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
% Define "Life" as the first cycle where discharge capacity drops below 1.75 Ah
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Life definition (threshold-based, dataset #3 uses 1.75 Ah)
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
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
Max_Out = [0.72, 1800, 0.9,  0.8,  0.75, 0.4];
Min_Out = [0.57, 400,  0.74, 0.15, 0,    0];

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
    Max_In = [4,    0.16;
              3.91, 0.17;
              3.83, 0.17;
              3.73, 0.18;
              3.6,  0.17;
              3.5,  0.17;
              3.35, 0.17;
              3.26, 0.17;
              3.17, 0.17;
              3.08, 0.155;
              2.99, 0.14;
              2.94, 0.13;
              2.9,  0.11];

    Min_In = [3.65, 0.05;
              3.55, 0.05;
              3.45, 0.05;
              3.35, 0.06;
              3.23, 0.05;
              3.17, 0.06;
              3.1,  0.07;
              3.03, 0.07;
              2.98, 0.07;
              2.93, 0.08;
              2.89, 0.08;
              2.84, 0.04;
              2.78, 0.02];

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
        Feature_Nor(:,i) = (F_Temp-F_Sort(40))/(F_Sort(end-40)-F_Sort(40));
    end

    % for j = 1:12
    %     figure(j),clf,plot(Feature_Nor(:,j))
    %     axis([0,433,0,1])
    % end

    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out evaluation (100 repeats)
    % For Dataset #3, exclude same-channel samples from the training set
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Build initial leave-one-out split
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Identify samples from the same tester channel as the held-out sample
            % (used to construct a cross-channel training set)
            CountSamChannel = 0;
            for IndDataCn = 1:length(Feature_Nor)
                if OneCycle(IndDataCn).Channel==OneCycle(IndexData).Channel
                    CountSamChannel = CountSamChannel+1;
                    SamChannelIndx(CountSamChannel) = IndDataCn;
                end
            end

            % Exclude same-channel samples from training to avoid leakage
            Temp_F(SamChannelIndx,:) = [];
            Temp_O(SamChannelIndx,:) = [];

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
    currentFile = sprintf('PLSR_Result_3_50_Y_Test_%d_5RC.mat',CountSV);
    save(currentFile,'Y_Test')
end