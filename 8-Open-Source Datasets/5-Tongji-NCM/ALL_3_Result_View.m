clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% This script: Merge RMSE statistics from two repeated experiments
%%% using weighted averaging, then visualize the
%%% distribution of RMSE values (per task) with violin plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load RMSE results
load("Result_View_RMSE_25.mat")

% Store intermediate results
T_RMSE_Mean       = RMSE_Mean;        % [model × task] mean RMSE
T_RMSE_Mean_Mean  = RMSE_Mean_Mean;   % [1 × task] overall mean RMSE

%% Load RMSE results
load("Result_View_RMSE_50.mat")

% Weighted averaging:
% 25-run experiment contributes 23 effective samples
% 50-run experiment contributes 28 effective samples
% Total effective sample size = 51
T_RMSE_Mean      = (T_RMSE_Mean*23 + RMSE_Mean*28) / 51;
T_RMSE_Mean_Mean = (T_RMSE_Mean_Mean*23 + RMSE_Mean_Mean*28) / 51;

%% Visualization (per task)
% For each of the 6 prediction tasks:
% - Violin plot shows RMSE distribution across models
% - Circle marker shows overall mean RMSE
% - Diamond marker shows RMSE of model #2 (baseline reference)
for i = 1:6
    figure, hold on, box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i),'o')
    plot(RMSE_Mean(2,i),'d')
end