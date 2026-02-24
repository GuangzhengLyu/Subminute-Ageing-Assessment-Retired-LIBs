clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Construct normalized health-indicator targets from the
%%% SDU-NCM single-cycle dataset, load prediction results from multiple
%%% model runs (Result_i_Y_Test_14.mat), compute per-task RMSE statistics,
%%% visualize RMSE distributions via violin plots, and save aggregated RMSE
%%% summaries for downstream comparison.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load SDU-NCM structured single-cycle dataset
load('../OneCycle_SDU_NCM_P1.mat')   

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life definition: full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1)    = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1)= OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Normalize to health-indicator scale (reference-based normalization)
Capa      = Capa/2.4;
Life      = Life;
ERate     = ERate/0.95;
CoChRate  = CoChRate/0.9;
MindVolt  = (MindVolt-2.65)/(3.65-2.65);
PlatfCapa = PlatfCapa/1.8;

%% Results aggregation (RMSE computation)
for i = 1:14
    % Load prediction results from model i
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Compute RMSE for each output dimension across repeats
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j)      = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE across repeats for model i (per task)
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-file temporary variables (keep aggregation arrays)
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

% Overall mean RMSE across all models (per task)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Relative performance indicator for model 2 (kept as in original script)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization: RMSE distributions across models (per task)
for i = 1:6
    figure,hold on,box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end

% Save aggregated RMSE summaries
save Result_View_RMSE_P1.mat RMSE_Mean RMSE_Mean_Mean