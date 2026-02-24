clear
close all
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Relaxation-Voltage Feature Extraction
%%% File: PLSR_LOOCV_SCU3_Dataset1_5RC.m
%%% Purpose:
%%% - Load SCU3 Dataset #1 and pre-extracted 5RC features (Uoc, R0, R1–R5, C1–C5)
%%% - Build targets: capacity SOH, RUL proxy (Life), and four expanded health indicators
%%% - Normalize targets to [0,1] with fixed bounds, clip outliers
%%% - Normalize features: (1) Uoc & R0 via setpoint-dependent min/max, (2) others via robust scaling
%%% - Train PLSR with leave-one-out evaluation, repeated 100 times, for each setpoint (13→1)
%%% Output:
%%% - Saves Y_Test for each setpoint to: PLSR_Result_1_70_Y_Test_%d_5RC.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
load('../../OneCycle_1.mat')
load('Feature_1_ALL_5RC.mat')

% Pack per-setpoint RC features into a single tensor: Feature(sample, setpoint, feature_id)
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
Feature(:,:,11) = R5;
Feature(:,:,12) = C5;

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

%% Convert to normalized health indicators (engineering-scale normalization)
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization bounds (fixed) and clipping to [0,1]
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

%% PLSR (per setpoint: 13 → 1)
for CountSV = 13:-1:1
    %% Feature normalization (setpoint-dependent)
    % Columns: [Uoc, R0]
    Max_In = [4.09, 0.05;
              4.005,0.048;
              3.92, 0.048;
              3.815,0.045;
              3.73, 0.05;
              3.63, 0.05;
              3.51, 0.055;
              3.45, 0.07;
              3.26, 0.075;
              3.2,  0.075;
              3.2,  0.075;
              3.1,  0.08;
              3.05, 0.08];

    Min_In = [4.04, 0.03;
              3.94, 0.03;
              3.86, 0.03;
              3.75, 0.03;
              3.65, 0.03;
              3.54, 0.03;
              3.44, 0.03;
              3.3,  0.04;
              3.2,  0.045;
              3.09, 0.05;
              3,    0.05;
              2.9,  0;
              2.8,  0];

    MyInd = 14-CountSV;

    % Normalize Uoc and R0 by predefined bounds
    Feature_Nor(:,1) = (Feature(:,CountSV,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,CountSV,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));

    % Robust normalization for remaining RC parameters (R1–R5, C1–C5)
    % - Use positive values only to build scaling reference
    % - Use 10th and (end-10)th sorted values to reduce outlier impact
    for i = 3:12
        F_Temp = Feature(:,CountSV,i);
        F_Sort = sort(F_Temp(find(F_Temp>0)));
        Feature_Nor(:,i) = (F_Temp-F_Sort(10))/(F_Sort(end-10)-F_Sort(10));
    end

    % for j = 1:12
    %     figure(j),clf,plot(Feature_Nor(:,j))
    %     axis([0,100,0,1])
    % end

    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    % Repeated leave-one-out evaluation (100 runs)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountSV
            CountRP
            IndexData

            % Build LOOCV split
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

            % Train PLSR (fixed to <= 3 latent components)
            ncomp = min(3, size(In_Train,1));
            [~, ~, ~, ~, beta] = plsregress(In_Train', Out_Train', ncomp);

            % Predict
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

        % Denormalize outputs to physical/engineering scales
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

%         % Optional visualization (train/test parity plots)
%         for IndOTrain = 1:6
%             figure(IndOTrain),hold on
%             plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
%
%             figure(IndOTrain+6),hold on
%             plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
%         end
    end

    % Save predictions for this setpoint
    currentFile = sprintf('PLSR_Result_1_70_Y_Test_%d_5RC.mat',CountSV);
    save(currentFile,'Y_Test')
end