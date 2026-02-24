clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% Description:
%%%   Load RMSE statistics from two experimental settings,
%%%   compute a weighted average of RMSE_Mean and RMSE_Mean_Mean according
%%%   to the sample counts (19 and 28, total = 47), and visualize the RMSE
%%%   distribution of each output using violin plots.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load RMSE results
load("Result_View_RMSE_25.mat")

T_RMSE_Mean = RMSE_Mean;
T_RMSE_Mean_Mean = RMSE_Mean_Mean;

% Load RMSE results
load("Result_View_RMSE_50.mat")

% Weighted fusion of RMSE statistics (19 + 28 = 47 total samples)
T_RMSE_Mean = (T_RMSE_Mean*19 + RMSE_Mean*28) / 47;
T_RMSE_Mean_Mean = (T_RMSE_Mean_Mean*19 + RMSE_Mean_Mean*28) / 47;

% Visualization of RMSE distribution per indicator
for i = 1:6
    figure,hold on,box on

    % Violin plot of RMSE across repeated runs
    violinplot(RMSE_Mean(:,i))

    % Mean RMSE across all runs
    plot(RMSE_Mean_Mean(i),'o')

    % RMSE of the second configuration (for reference)
    plot(RMSE_Mean(2,i),'d')
end