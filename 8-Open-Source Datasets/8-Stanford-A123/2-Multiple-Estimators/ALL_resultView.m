clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 8-Stanford-A123
%%% This script: Load the single-cycle dataset and multi-model prediction
%%% results, compute RMSE distributions for each health indicator across
%%% repeated runs, and visualize the RMSE statistics using violin plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load('../OneCycle_MITA123_4.mat')

%% Sample construction (ground-truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: full discharge-capacity trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators (kept as in the original script)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health indicators (dataset-specific scaling)
% Note: Keep the original scaling factors to ensure consistent evaluation.
Capa = Capa/1.1;
Life = Life;
ERate = ERate/0.89;
CoChRate = CoChRate/0.99;
MindVolt = (MindVolt-2)/(3.12-2);
PlatfCapa = PlatfCapa/0.97;

%% Result evaluation (RMSE)
% Loop over result files, each containing Y_Test for a given model index i.
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Compute RMSE for each repeated run (j) and each output dimension
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE per model (averaged across runs)
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-iteration variables to reduce workspace contamination
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

%% Summary statistics across models
% Mean RMSE across all models for each output dimension
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative RMSE ratio of model i=2 compared to the across-model mean (averaged over 6 outputs)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization (violin plots)
% For each output, show the RMSE distribution across models,
% and mark (1) the across-model mean and (2) the model i=2 reference point.
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end