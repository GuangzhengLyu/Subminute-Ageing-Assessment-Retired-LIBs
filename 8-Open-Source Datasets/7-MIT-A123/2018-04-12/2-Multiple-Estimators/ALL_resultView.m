clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123 (MITA123_3)
%%% This script: Load ground-truth health indicators, read multi-model
%%% prediction results, compute RMSE distributions and summary statistics,
%%% visualize RMSE via violin plots, and save aggregated RMSE metrics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load('../OneCycle_MITA123_3.mat')   

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life: total number of available discharge-capacity samples (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to normalized health indicators (dataset-specific scaling)
% Note: These scalings are preserved exactly to keep consistency with the
% original evaluation protocol.
Capa = Capa/1.1;
Life = Life;
ERate = ERate/0.9;
CoChRate = CoChRate/0.995;
MindVolt = (MindVolt-2)/(3.12-2);
PlatfCapa = PlatfCapa/0.97;

%% Result summary
% For each model (i = 1:14), load the saved prediction tensor Y_Test and
% compute RMSE across repeated runs (j indexes repeated trials).
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

    % Mean RMSE over repeated runs for each output dimension
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-model temporary arrays to avoid carry-over
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

% Overall mean RMSE across models (per output dimension)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative performance of model #2 compared with the overall mean (scalar)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization: RMSE distributions across models
% For each output dimension, plot violin distribution across models and
% overlay the overall mean and model #2 mean for reference.
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end

% Save aggregated RMSE results
save Result_View_RMSE_3.mat RMSE_Mean RMSE_Mean_Mean