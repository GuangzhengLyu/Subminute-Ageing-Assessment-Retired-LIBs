clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 2-RWTH-NCA
%%% This script: Build normalized multi-dimensional health indicators from
%%% the RWTH-NCA structured dataset, load repeated prediction results from
%%% multiple baseline models, compute RMSE distributions for each task, and
%%% visualize per-model RMSE statistics using violin plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset (RWTH-NCA)
load('../OneCycle_NCA_RWTH.mat')

%% Sample construction
% Build ground-truth outputs from cycle-level records
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: use the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to health-indicator scales
% Keep the same scaling/normalization conventions as used in the modelling scripts
Capa = Capa/13000;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.7-3);
PlatfCapa = PlatfCapa/5000;

%% Result evaluation
% For each baseline model result file: compute RMSE per run and summarize across runs
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j)      = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Per-model mean RMSE across repeated runs
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-file temporary variables (kept as in original script)
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

% Overall mean RMSE across all models (per task)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Example aggregate ratio (as in original script): model #2 vs overall mean (averaged across tasks)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization
% Violin plots of per-model RMSE distributions (one plot per task)
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end