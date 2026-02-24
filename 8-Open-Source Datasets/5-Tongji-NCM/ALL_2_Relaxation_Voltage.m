close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% This script: Extract relaxation-voltage segments (near end-of-charge /
%%% high-voltage region with zero current) for each cell, then visualize
%%% relaxation voltage trajectories across all cells after sorting samples
%%% by initial discharge capacity (OrigCapaAh). Line colors encode ageing
%%% severity (default: early-ageing colormap; deep-ageing option provided).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Extract relaxation-voltage segments (zero current + high voltage region)
for IndexData = 1:length(OneCycle)

    % Identify relaxation window: I = 0 and V >= 4.0 V
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV >= 4);
    IndexRX = intersect(Index1,Index2);

    % Extract the relaxation-voltage segment (include the sample right before)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

%% Sort cells by initial discharge capacity (OrigCapaAh)
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A, B] = sort(MyCapa); %#ok<ASGLU>

%% Plot relaxation-voltage segments for each cell (sorted order)
for i = 1:NumData
    Index = B(i);

    %% Early-ageing color mapping (higher initial capacity -> darker tone)
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep-ageing color mapping option (based on cycle life)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    % Relaxation voltage (RV) sequence for plotting
    RV{Index,1} = Vrlx{Index,1};

    % Overlay all cells in one figure
    figure(1), hold on, box on, plot(RV{Index,1},'-','color',ColorMap);
end