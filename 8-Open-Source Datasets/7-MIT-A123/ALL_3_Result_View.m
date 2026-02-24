%%% ========================================================================
%%% Project : External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset : 7-MIT-A123 (Cross-dataset RMSE aggregation)
%%% This script: Load RMSE statistics from Dataset #2 and Dataset #3,
%%%              compute weighted averages, and visualize distributions
%%%              using violin plots.
%%% ========================================================================

clear
clc

%% Load RMSE results from Dataset #2
load("Result_View_RMSE_2.mat")

T_RMSE_Mean = RMSE_Mean;
T_RMSE_Mean_Mean = RMSE_Mean_Mean;

%% Load RMSE results from Dataset #3 and compute weighted average
load("Result_View_RMSE_3.mat")

% Weighted aggregation (43 samples in Dataset #2, 46 in Dataset #3)
T_RMSE_Mean = (T_RMSE_Mean*43 + RMSE_Mean*46) / 89;
T_RMSE_Mean_Mean = (T_RMSE_Mean_Mean*43 + RMSE_Mean_Mean*46) / 89;

%% Visualization (per task)
% For each of the 6 output tasks:
% - Violin plot of RMSE across models
% - Circle marker: overall mean RMSE
% - Diamond marker: RMSE of Model #2 (example reference model)

for i = 1:6
    figure, hold on, box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end