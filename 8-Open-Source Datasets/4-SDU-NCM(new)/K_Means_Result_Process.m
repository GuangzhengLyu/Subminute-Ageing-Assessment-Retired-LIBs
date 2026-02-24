clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Merge clustering-consistency results (K-means based metric)
%%% across multiple proto batches (P1/P2/...) using weighted averaging, then
%%% visualize the aggregated ratios with bar charts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("K_Means_Result_P1.mat")
for i = 1:5
    % P1 weight = 8
    MyResult(i,:) = Result(i,:) .* 8;
end

load("K_Means_Result_P2.mat")
for i = 1:5
    % P2 weight = 8
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 8;
end

load("K_Means_Result_P3.mat")
for i = 1:5
    % P3 weight = 6
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 6;
end

load("K_Means_Result_P4.mat")
for i = 1:5
    % P4 weight = 8
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 8;
end

load("K_Means_Result_P5.mat")
for i = 1:5
    % P5 weight = 8
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 8;
end

load("K_Means_Result_P6.mat")
for i = 1:5
    % P6 weight = 6
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 6;
end

load("K_Means_Result_P15.mat")
for i = 1:5
    % P15 weight = 7
    MyResult(i,:) = MyResult(i,:) + Result(i,:) .* 7;
end

% Normalize by total weight: 8+8+6+8+8+6+7 = 51
for i = 1:5
    MyResult(i,:) = MyResult(i,:) ./ 51;
end

% Visualization: each row (metric) plotted as a bar chart over Kind (columns)
figure(1), clf, bar(MyResult(1,:))
figure(2), clf, bar(MyResult(2,:))
figure(3), clf, bar(MyResult(3,:))
figure(4), clf, bar(MyResult(4,:))
figure(5), clf, bar(MyResult(5,:))