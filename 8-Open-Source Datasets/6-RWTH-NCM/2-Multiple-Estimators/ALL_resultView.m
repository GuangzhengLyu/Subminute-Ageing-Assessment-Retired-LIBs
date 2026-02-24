clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Assemble ground-truth health indicators from the RWTH-NCM
%%% single-cycle structure, load prediction results from multiple models,
%%% compute RMSE statistics across repetitions, and visualize the RMSE
%%% distributions using violin plots with mean/baseline markers.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RWTH-NCM dataset
load('../OneCycle_NCM_RWTH.mat')

%% Sample construction
% Build per-sample ground-truth targets from the cycling trajectories
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health-indicator targets
% Apply dataset-specific scaling to map each indicator into a comparable range
Capa = Capa/2;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.9-3);
PlatfCapa = PlatfCapa/1.5;

%% Result summary (RMSE across models and repetitions)
% For each model index i, load its stored Y_Test and compute RMSE per output
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE for this model across repetitions
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-model temporary variables to avoid cross-iteration contamination
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

%% Aggregate summaries
% Overall mean RMSE across models
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative performance of the second model vs the overall mean (as in original script)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization: violin plots of RMSE distributions across models
% Each figure shows the RMSE distribution across the 14 models for one output
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end