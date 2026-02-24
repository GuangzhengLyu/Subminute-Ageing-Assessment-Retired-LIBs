clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA (TongjiNCA25)
%%% This script: Summarize repeated-test prediction errors across multiple
%%% result files by computing per-output RMSE distributions, visualizing
%%% RMSE spread with violin plots, and saving aggregated RMSE statistics.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tongji-NCA dataset (structured single-cycle data)
load('../OneCycle_TongjiNCA25.mat')  

%% Sample construction
% Build sample-level targets: life, original capacity, and expanded health
% indicators (multi-dimensional outputs).
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
% used during PLSR training/prediction.
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.92;
CoChRate = CoChRate/0.83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.35;

%% Results summary (RMSE aggregation)
% Loop over result files Result_i_Y_Test_14.mat, compute RMSE for each output,
% then store mean RMSE per file and per output dimension.
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Per-repetition RMSE (j indexes repetitions in Y_Test)
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE across repetitions (reported per output dimension)
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-file buffers (kept as in original script)
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

%% Aggregate statistics and visualization
% RMSE_Mean_Mean: mean RMSE across the 14 result files (per output dimension)
% RMSE_Mean_Percent: relative ratio of the 2nd file to the overall mean (scalar)
RMSE_Mean_Mean = mean(RMSE_Mean,1);
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

% Violin plots of RMSE distributions across result files (per output dimension)
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end

% Save aggregated RMSE statistics for downstream reporting
save Result_View_RMSE_25.mat RMSE_Mean RMSE_Mean_Mean