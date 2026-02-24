clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Merge RMSE summaries from multiple proto batches (P1/P2/...)
%%% using weighted averaging (weights = sample counts per batch), then plot
%%% violin distributions and reference markers.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("Result_View_RMSE_P1.mat")

% Weighted accumulation (P1 weight = 8)
T_RMSE_Mean = RMSE_Mean * 8;
T_RMSE_Mean_Mean = RMSE_Mean_Mean * 8;

load("Result_View_RMSE_P2.mat")

% P2 weight = 8
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 8;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 8;

load("Result_View_RMSE_P3.mat")

% P3 weight = 6
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 6;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 6;

load("Result_View_RMSE_P4.mat")

% P4 weight = 8
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 8;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 8;

load("Result_View_RMSE_P5.mat")

% P5 weight = 8
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 8;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 8;

load("Result_View_RMSE_P6.mat")

% P6 weight = 6
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 6;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 6;

load("Result_View_RMSE_P15.mat")

% P15 weight = 7
T_RMSE_Mean = T_RMSE_Mean + RMSE_Mean * 7;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean + RMSE_Mean_Mean * 7;

% Normalize by total weight: 8+8+6+8+8+6+7 = 51
T_RMSE_Mean = T_RMSE_Mean / 51;
T_RMSE_Mean_Mean = T_RMSE_Mean_Mean / 51;

% Visualization:
% - NOTE: Current plotting uses RMSE_Mean / RMSE_Mean_Mean from the *last loaded*
%   file (P15). If you intend to visualize the aggregated results, replace
%   RMSE_Mean -> T_RMSE_Mean and RMSE_Mean_Mean -> T_RMSE_Mean_Mean.
for i = 1:6
    figure, hold on, box on
    violinplot(RMSE_Mean(:,i))
    plot(RMSE_Mean_Mean(i), 'o')
    plot(RMSE_Mean(2,i), 'd')
end