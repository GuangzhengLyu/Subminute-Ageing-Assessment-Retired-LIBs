clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 5-Tongji-NCM
%%% This script: Load ground-truth health indicators from the Tongji-NCM
%%% OneCycle data, evaluate RMSE for multiple model result files (1–14),
%%% summarize mean RMSE across repetitions, and visualize distributions.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset (Tongji-NCM)
load('../OneCycle_TongjiNCM.mat')  

%% Sample construction
% Assemble ground-truth targets (life, capacity, and expanded indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life defined as the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to health-indicator scale (reference normalization)
% Keep the original operations; only clarify intent via comments
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.93;
CoChRate = CoChRate/0.86;
MindVolt = (MindVolt-2.65)/(3.50-2.65);
PlatfCapa = PlatfCapa/1.31;

%% Result summary (RMSE evaluation across result files)
% Each "Result_%d_Y_Test_14.mat" is expected to contain Y_Test (repetitions x 6 x N)
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Compute RMSE for each repetition j and each task (6 outputs)
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE across repetitions for each task
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear temporary variables to avoid accidental reuse across files
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

% Overall mean RMSE across all evaluated models (1–14)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative RMSE of model #2 compared to the overall mean (averaged across tasks)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization: violin plots of RMSE distributions across models
% For each task, show the distribution of RMSE_Mean across models
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end

% Save summarized RMSE results for downstream comparison/plotting
save Result_View_RMSE_50.mat RMSE_Mean RMSE_Mean_Mean