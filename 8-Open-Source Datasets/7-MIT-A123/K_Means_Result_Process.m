%%% ========================================================================
%%% Project : External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset : 7-MIT-A123 (Cross-dataset clustering evaluation)
%%% This script: Load K-means-based evaluation results from Dataset #2 and
%%%              Dataset #3, compute weighted averages, and visualize them
%%%              using bar charts.
%%% ========================================================================

clear
clc

%% Load K-means evaluation results from Dataset #2
load("K_Means_Result_2.mat")
for i = 1:5
    MyResult(i,:) = Result(i,:) .* 43;   % Weight by 43 samples
end

%% Load K-means evaluation results from Dataset #3
load("K_Means_Result_3.mat")
for i = 1:5
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 46;  % Add weighted results (46 samples)
end

%% Compute weighted average (total samples = 43 + 46 = 89)
for i = 1:5
    MyResult(i,:) = MyResult(i,:) ./ 89;
end

%% Visualization (bar charts for each evaluation metric)
figure(1), clf, bar(MyResult(1,:))
figure(2), clf, bar(MyResult(2,:))
figure(3), clf, bar(MyResult(3,:))
figure(4), clf, bar(MyResult(4,:))
figure(5), clf, bar(MyResult(5,:))