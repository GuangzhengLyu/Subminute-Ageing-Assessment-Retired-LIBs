clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% Description:
%%%   Load K-means evaluation results from two experimental settings
%%%   , compute the weighted average according
%%%   to the sample counts (19 and 28, total = 47), and visualize the
%%%   averaged clustering performance metrics using bar charts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load clustering results
load("K_Means_Result_25.mat")

% Initialize weighted accumulation (weight = 19)
for i = 1:5
    MyResult(i,:) = Result(i,:) .* 19;
end

% Load clustering results
load("K_Means_Result_50.mat")

% Accumulate weighted results (weight = 28)
for i = 1:5
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 28;
end

% Normalize by total sample count (19 + 28 = 47)
for i = 1:5
    MyResult(i,:) = MyResult(i,:) ./ 47;
end

% Visualization: bar plot for each clustering metric
figure(1),clf,bar(MyResult(1,:))
figure(2),clf,bar(MyResult(2,:))
figure(3),clf,bar(MyResult(3,:))
figure(4),clf,bar(MyResult(4,:))
figure(5),clf,bar(MyResult(5,:))