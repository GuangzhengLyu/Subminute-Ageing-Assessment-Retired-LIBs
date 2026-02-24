close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 9-SCU1-A123
%%% This script: Extract and visualize relaxation-voltage (RV) segments.
%%% For each cell, the RV segment is taken from the sample immediately
%%% before the relaxation step (Steps == 2) through the end of that step.
%%% Cells are then sorted by initial capacity and plotted with a
%%% capacity-ordered colormap to highlight early ageing trends.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract relaxation-voltage (RV) segments for each cell
for IndexData = 1:length(OneCycle)

    % Identify indices corresponding to the relaxation step
    IndexRX = find(OneCycle(IndexData).Steps == 2);

    % Extract voltage from the sample right before relaxation starts
    % through the end of the relaxation step
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

% Number of cells (samples)
NumData = length(OneCycle);

% Collect initial (original) capacity for sorting
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells by initial capacity (ascending)
[A, B] = sort(MyCapa);

for i = 1:NumData
    % Map sorted order back to the original sample index
    Index = B(i);

    %% Early ageing colormap (capacity-ranked)
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep ageing colormap (alternative; disabled in the original code)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    % Alias for clarity: RV is the extracted relaxation-voltage segment
    RV{Index,1} = Vrlx{Index,1};

    % Plot RV trajectories (each cell as one curve)
    figure(1), hold on, box on, plot(RV{Index,1},'-','color',ColorMap);
end