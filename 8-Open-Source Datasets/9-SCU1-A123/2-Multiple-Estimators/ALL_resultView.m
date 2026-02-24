clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 9-SCU1-A123
%%% This script: Aggregate prediction results (Y_Test) from multiple models,
%%% compute RMSE for capacity/life and extended health indicators, then
%%% summarize distributions across models with violin plots and reference
%%% markers (overall mean and a selected baseline model).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load('../OneCycle_A123.mat')

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Use full trajectory length as the life label (as in the original script)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators
    ERate(CountData,1)    = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1)= OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health indicators (manual scaling to comparable ranges)
Capa = Capa/1.2;
Life = Life;
ERate = ERate/81;
CoChRate = CoChRate/54;
MindVolt = (MindVolt-2)/(2.93-2);
PlatfCapa = PlatfCapa./0.8;

%% Result aggregation and RMSE evaluation
% Loop over result files (e.g., multiple models/variants indexed by i)
for i = 1:14
    currentFile = sprintf('Result_%d_Y_Test_14.mat',i);
    load(currentFile)

    % Each repetition j provides one prediction tensor Y_Test(j, outputDim, sample)
    for j = 1:size(Y_Test,1)
        RMSE_Capa(i,j)      = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end

    % Mean RMSE across repetitions for each output dimension
    RMSE_Mean(i,1) = mean(RMSE_Capa(i,:));
    RMSE_Mean(i,2) = mean(RMSE_Life(i,:));
    RMSE_Mean(i,3) = mean(RMSE_ERate(i,:));
    RMSE_Mean(i,4) = mean(RMSE_CoChR(i,:));
    RMSE_Mean(i,5) = mean(RMSE_MipV(i,:));
    RMSE_Mean(i,6) = mean(RMSE_PlatfCapa(i,:));

    % Clear per-model temporary variables (kept as in original script)
    clear RMSE_Capa RMSE_Life RMSE_ERate RMSE_CoChR RMSE_MipV RMSE_PlatfCapa Y_Test
end

% Overall mean RMSE across all models (per output dimension)
RMSE_Mean_Mean = mean(RMSE_Mean,1);

% Reference ratio: model #2 relative to the overall mean (averaged across outputs)
RMSE_Mean_Percent = mean(RMSE_Mean(2,:)./RMSE_Mean_Mean);

%% Visualization: violin plots of RMSE distributions across models
for i = 1:6
    figure,hold on,box on

    % RMSE distribution across models for output dimension i
    violinplot(RMSE_Mean(:,i))

    % Markers for quick comparison:
    %   - circle: overall mean across models
    %   - diamond: selected reference model (#2)
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end