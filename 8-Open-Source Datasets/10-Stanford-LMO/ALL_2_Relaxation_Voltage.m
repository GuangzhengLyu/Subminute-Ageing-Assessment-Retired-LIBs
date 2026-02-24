clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 10-Stanford-LMO
%%% This script: Extract relaxation-voltage segments (Vrlx) for each cell
%%% using a simple rule (I = 0 and V >= 3.7 V), then plot the relaxation
%%% voltage trajectories sorted by initial capacity, with a life-dependent
%%% colormap to highlight deep-ageing differences.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("OneCycle_Stanford_LMO.mat")

%% Extract relaxation-voltage segment for each sample
for IndexData = 1:length(OneCycle)

    % Candidate indices where current is zero (relaxation period)
    Index1 = find(OneCycle(IndexData).CurrentA == 0);

    % Candidate indices where voltage is above the threshold
    Index2 = find(OneCycle(IndexData).VoltageV >= 3.7);

    % Intersection gives the relaxation segment indices under the voltage condition
    IndexRX = intersect(Index1,Index2);

    % Extract the relaxation-voltage trace (include one sample before the first point)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

%% Sort cells by initial capacity for consistent visualization
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

[A B] = sort(MyCapa);

%% Plot relaxation-voltage traces with ageing-dependent colormap
for i = 1:NumData
    Index = B(i);

    %% Early ageing (optional colormap design)
    % ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing (life-dependent colormap design)
    % Life is defined here as the length of the discharge-capacity trajectory
    Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    ColorMap = [1 0.4+0.6*(Life-1200)/2100 0.2-0.2*(Life-1200)/2100];

    % Relaxation-voltage trace for the selected cell
    RV{Index,1} = Vrlx{Index,1};
    
    figure(1),hold on,box on,plot(RV{Index,1},'-','color',ColorMap);
end