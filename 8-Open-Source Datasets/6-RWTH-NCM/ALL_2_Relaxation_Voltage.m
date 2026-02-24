clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Extract relaxation-voltage segments from each sample using
%%% a zero-current and high-voltage gate, keep only the last continuous
%%% relaxation block, and visualize the relaxation traces. Samples are
%%% sorted by original capacity and color-coded to highlight heterogeneity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load("OneCycle_NCM_RWTH.mat")

%% Relaxation-voltage extraction for each sample
for IndexData = 1:length(OneCycle)

    % Identify relaxation points: zero current and high-voltage region
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV > 4.05);
    IndexRX = intersect(Index1,Index2);

    % Keep only the last continuous relaxation segment (remove earlier blocks)
    % A gap larger than 10 samples indicates a discontinuity in the index series
    TempIndex = IndexRX;
    for i = length(IndexRX):-1:2
        if IndexRX(i)-IndexRX(i-1) > 10
            TempIndex(1:i) = [];
        end
    end
    IndexRX = TempIndex;

    % Store relaxation voltage trace (includes one point before the segment start)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

%% Sort samples by original capacity (ascending)
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A B] = sort(MyCapa);

%% Plot relaxation traces (color-coded by sorted capacity rank)
for i = 1:NumData
    Index = B(i);

    %% Early ageing color map (capacity-ranked)
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing color map (optional; keep the original commented logic)
    % if min(OneCycle(Index).Cycle.DiscCapaAh)<=2.1
    %     Life = min(find(OneCycle(Index).Cycle.DiscCapaAh <= 2.1));
    % else
    %     Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % end
    % ColorMap = [1 0.4+0.6*(Life-200)/605 0.2-0.2*(Life-200)/605];

    % Use the extracted relaxation voltage trace
    RV{Index,1} = Vrlx{Index,1};

    % Plot relaxation voltage trace
    figure(1),hold on,box on,plot(RV{Index,1},'-','color',ColorMap);
end