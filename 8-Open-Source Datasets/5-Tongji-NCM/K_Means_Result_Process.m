clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% This script: Merge K-means-based consistency metrics from two repeated
%%% experiments using weighted averaging, then
%%% visualize the aggregated results for five evaluation indicators.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load K-means results
load("K_Means_Result_25.mat")

% Weighted accumulation (23 effective samples from 25-run experiment)
for i = 1:5
    MyResult(i,:) = Result(i,:) .* 23;
end

%% Load K-means results
load("K_Means_Result_50.mat")

% Add weighted contribution (28 effective samples from 50-run experiment)
for i = 1:5
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 28;
end

%% Final weighted average (total effective sample size = 51)
for i = 1:5
    MyResult(i,:) = MyResult(i,:) ./ 51;
end

%% Visualization (bar plots for five indicators)
figure(1), clf, bar(MyResult(1,:))
figure(2), clf, bar(MyResult(2,:))
figure(3), clf, bar(MyResult(3,:))
figure(4), clf, bar(MyResult(4,:))
figure(5), clf, bar(MyResult(5,:))