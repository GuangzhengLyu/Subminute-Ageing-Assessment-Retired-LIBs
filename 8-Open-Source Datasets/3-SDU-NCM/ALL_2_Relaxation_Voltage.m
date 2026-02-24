close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 3-SDU-NCM
%%% This script: Extract relaxation-voltage segments (zero current, high
%%% terminal-voltage region) for each sample and visualize the relaxation
%%% voltage traces. Cells are sorted by original capacity, and curve colors
%%% are assigned as a function of cycle-life length (deep-ageing view).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Relaxation-voltage extraction (per sample)
for IndexData = 1:length(OneCycle)

    % Identify relaxation points: zero current and high voltage region
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV >= 4.1);
    IndexRX = intersect(Index1,Index2);

    % Store relaxation voltage trace (keep original indexing rule)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

%% Sort cells by original capacity (ascending) for visualization
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A B] = sort(MyCapa); %#ok<ASGLU>

for i = 1:NumData
    Index = B(i);

    %% Early-ageing color option (kept as comment in original script)
    % ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep-ageing color mapping (use life length to control color)
    Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    ColorMap = [1 0.4+0.6*(Life-190)/2000 0.2-0.2*(Life-190)/2000];

    % Alias for clarity (kept as a separate variable in the original script)
    RV{Index,1} = Vrlx{Index,1};

    % Plot relaxation voltage traces (each cell as one curve)
    figure(1),hold on,box on,plot(RV{Index,1},'-','color',ColorMap);
end