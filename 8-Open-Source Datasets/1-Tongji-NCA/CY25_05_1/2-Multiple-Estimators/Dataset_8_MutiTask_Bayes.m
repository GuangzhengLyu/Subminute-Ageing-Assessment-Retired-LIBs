clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA (TongjiNCA25)
%%% This script: Assemble relaxation-feature tensors and normalized outputs,
%%% then run repeated leave-one-out evaluation using a Bayesian linear
%%% regression model. The model is trained on normalized inputs/outputs and
%%% predictions are de-normalized back to the original scales. Test-set
%%% predictions are saved to a MAT-file for downstream analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_TongjiNCA25.mat')  
load('../Feature_ALL_TongjiNCA25.mat')

%% Feature tensor assembly
% Feature(:,:,k) stores the k-th relaxation-derived parameter across
% samples (row) and voltage setpoints (column). For this dataset, features
% are provided as [N x 1] and placed into Feature(:,1,k).
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build sample-level targets: life, original capacity, and expanded health
% indicators (used as multi-dimensional outputs).
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition for this script: full available trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (cycle-index selection follows raw data structure)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unified health-indicator scaling
% Convert raw variables into comparable normalized health-indicator forms.
% Note: This is a unified engineering scaling, not the [0,1] normalization
% used during model fitting/prediction below.
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.92;
CoChRate = CoChRate/0.83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.35;

%% Output normalization
% Apply min-max normalization to construct the multi-dimensional output matrix.
% Output columns: [Capacity-based SOH, Life, EnergyRate, ConstCharRate, MindVolt, PlatfCapa]
Max_Out = [0.934, 200, 0.995, 0.98,  0.99, 0.97];
Min_Out = [0.92,  100, 0.965, 0.972, 0.95, 0.9];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0,1]
Output(Output<0) = 0;Output(Output>1) = 1;

%% Repeated LOOCV evaluation (Bayesian linear regression)
% For the selected setpoint index (CountSV), normalize input features using
% setpoint-specific bounds, then run repeated leave-one-out evaluation.
for CountSV = 14:-1:14 

    %% Input feature normalization (setpoint-dependent)
    % Max_In/Min_In provide per-setpoint bounds for the 6 relaxation features:
    % [Uoc, R0, R1, C1, R2, C2]
    Max_In = [4.159, 0.063, 0.09,  62, 0.09,  62];
    Min_In = [4.152, 0.053, 0.074, 54, 0.074, 54];

    % Map CountSV to row index in bound tables (kept as in original script)
    MyInd = 15-CountSV;
          
    % Min-max normalize the 6 relaxation features (stored at column index 1)
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized inputs/outputs to [0,1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    % Repeated leave-one-out evaluation (CountRP repetitions)
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            % Progress printing (kept as in original script)
            CountRP
            IndexData

            % Construct LOOCV split: one sample for test, remaining for training
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Ensure 2-D shapes are consistent for matrix operations in the helper functions
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% Parameter settings
            sigma_squared = 0.01; % Variance term for output covariance
            eta_sqaured   = 0.02; % Variance term for coefficient prior covariance

            %% Model training
            % Train Bayesian regression on the training set (kept as in original script)
            model = my_bayesian_regression(In_Train', Out_Train', sigma_squared, eta_sqaured);

            % Prediction (training and held-out test sample)
            Y_Train = my_sim_bayes(model,In_Train')';
            Y_Test(CountRP,:,IndexData) = my_sim_bayes(model,In_Test(:,IndexData)');
        
            toc
        end

        % Clip predictions to [0,1] in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        %% De-normalization
        % Convert predictions and labels back to the original output scales
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
        
        %% Result visualization
        % Scatter plots: ground truth vs prediction for training and test sets
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save repeated-test predictions for the current setpoint
    currentFile = sprintf('Result_8_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end

%% Functions
function mu = my_bayesian_regression(p_train, t_train, small_sigma_squared, eta_sqaured)
    % Get the feature dimension and sample count of the input matrix
    M = size(p_train, 1);
    N = size(p_train, 2);
    %
    big_sigma = small_sigma_squared * eye(M);
    big_omega = eta_sqaured * eye(N);
    % Compute posterior mean of coefficients
    lambda = p_train' / big_sigma * p_train + inv(big_omega);
    mu     = lambda \ p_train' / big_sigma * t_train;
end

function t_sim = my_sim_bayes(model, p_train)
    % Compute predicted outputs
    t_sim = p_train * model;
end