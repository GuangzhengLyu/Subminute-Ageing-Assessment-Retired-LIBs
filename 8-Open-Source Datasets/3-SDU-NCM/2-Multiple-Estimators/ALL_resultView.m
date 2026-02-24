clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 3-SDU-NCM
%%% This script: Load the SDU-NCM single-cycle dataset, construct normalized
%%% health indicators (capacity, life length, and expanded performance metrics),
%%% then summarize multi-task prediction errors across multiple result files.
%%% RMSE distributions are visualized using violin plots, with reference markers
%%% for the overall mean RMSE and a selected baseline method (index = 2).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset (ground truth targets)
% SCU3 dataset
load('../OneCycle_SDU_NCM.mat')

%% Sample construction (ground truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life: here defined as the available trajectory length (kept as original)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to normalized health indicators
% Normalize each indicator to a reference scale (kept as original)
Capa     = Capa/2.4;
Life     = Life;
ERate    = ERate/0.95;
CoChRate = CoChRate/0.9;
MindVolt = (MindVolt-2.65)/(3.65-2.65);
PlatfCapa= PlatfCapa/1.8;

%% Result summarization (RMSE across result files and repeats)
% Each file "Result_i_Y_Test_14.mat" is expected to contain Y_Test:
%   Y_Test(repeat, output-dim, sample)
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Compute RMSE per repeat for each output dimension
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE over repeats (one row per method i)
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-file temporary variables to reduce workspace clutter
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

%% Aggregate statistics and visualization
% Overall mean RMSE across all methods (one scalar per output dimension)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative RMSE of the selected baseline method (index = 2) vs overall mean
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

% Violin plots of RMSE across methods for each output dimension
for i = 1:6
    figure,hold on,box on

    % Distribution of per-method mean RMSE
    violinplot(RMSE_Mean(:,i))

    % Marker: overall mean RMSE across methods
    plot(RMSE_Mean_Mean(i),'o')

    % Marker: baseline method RMSE (method index = 2)
    plot(RMSE_Mean(2,i),'d')
end