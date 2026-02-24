clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 1-Tongji-NCA
%%% This script: Load Tongji-NCA ground-truth health indicators, apply the
%%% same indicator scaling used in the main workflow, then batch-evaluate
%%% prediction results stored in multiple MAT-files (Result_%d_Y_Test_14.mat).
%%% For each file, compute RMSE distributions over repeated runs, aggregate
%%% mean RMSE per task, visualize RMSE variability using violin plots, and
%%% save summary statistics for reporting.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tongji-NCA dataset
load('../OneCycle_TongjiNCA.mat')  

%% Sample construction (ground-truth labels)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full available discharge-capacity trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (as stored in OneCycle)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert raw measurements into normalized health indicators
% NOTE: The scaling factors below are kept exactly as in the original script
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.93;
CoChRate = CoChRate/0.86;
MindVolt = (MindVolt-2.65)/(3.54-2.65);
PlatfCapa = PlatfCapa/1.35;

%% Result summary: batch RMSE evaluation across saved prediction files
for i = 1:14
    % Each MAT-file is expected to contain Y_Test (runs × outputs × samples)
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Compute RMSE for each run j and each indicator
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE over runs for this file i (per task)
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-file intermediates to avoid carry-over
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

%% Aggregate statistics and visualization
% Mean of the per-file mean RMSE (baseline summary across i = 1..14)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Mean ratio of file #2 to the overall mean (kept as in original script)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

% Violin plots: RMSE distribution across files for each task
% NOTE: violinplot must exist on the MATLAB path
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end

% Save summary metrics for downstream reporting/plotting
save Result_View_RMSE_50.mat RMSE_Mean RMSE_Mean_Mean