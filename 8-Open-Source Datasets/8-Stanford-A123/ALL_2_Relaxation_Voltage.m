close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 8-Stanford-A123
%%% This script: Extract relaxation-voltage segments (Vrlx) from the
%%% selected cycle waveform and visualize RV trajectories across cells.
%%% Relaxation segments are identified by:
%%%   (1) zero current (I == 0), and
%%%   (2) voltage threshold (V >= 3.5 V),
%%% then taking the last continuous relaxation block and plotting it with a
%%% capacity-ranked colormap.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Extract relaxation-voltage segment for each cell
for IndexData = 1:length(OneCycle)

    % Identify candidate relaxation indices:
    % - Current equals zero (rest/relaxation)
    % - Voltage above threshold (avoid low-voltage region)
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV >= 3.5);
    IndexRX = intersect(Index1,Index2);

    % Keep the last continuous block in IndexRX
    % (If there are multiple rest blocks, use the final one.)
    TempIR = IndexRX;
    for i = 1:length(IndexRX)-1
        if IndexRX(i+1)-IndexRX(i) > 1
            TempIR = IndexRX(i+1:end);
        end
    end
    IndexRX = TempIR;

    % Extract relaxation voltage trace
    % Note: IndexRX(1)-1 is used to include the point immediately before rest.
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

%% Sort cells by initial capacity for visualization order
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A, B] = sort(MyCapa); % B: sorted indices by initial capacity

%% Plot relaxation-voltage (RV) trajectories in sorted order
for i = 1:NumData
    Index = B(i);

    %% Early-ageing colormap (ranked by initial capacity)
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep-ageing colormap (optional; based on cycle life)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    % Alias for plotting
    RV{Index,1} = Vrlx{Index,1};

    % Plot RV trace
    figure(1),hold on,box on
    plot(RV{Index,1}, '-', 'color', ColorMap);
end