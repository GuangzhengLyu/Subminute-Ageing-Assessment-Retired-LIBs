close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 8-Stanford-A123
%%% This script: Visualize per-cell ageing trajectories after robust
%%% outlier filtering (Hampel) and smoothing (Savitzky–Golay). Cells are
%%% sorted by initial capacity and plotted with a capacity-ranked colormap.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumData = length(OneCycle);

%% Collect and sort initial capacities (for ranking and color mapping)
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A, B] = sort(MyCapa); % A: sorted capacity values, B: sorted indices

%% Loop through cells in sorted order and plot smoothed trajectories
for i = 1:NumData
    Index = B(i);

    %% Early-ageing color map (ranked by initial capacity)
    % Higher initial capacity -> darker/stronger tone (as i increases, color fades)
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep-ageing color map (optional; based on cycle life)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    %% Filtering/smoothing configuration
    % Note: FilterLength should be odd for sgolayfilt.
    FilterLength = 51;

    %% Capacity trajectory (discharge capacity)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);                         % robust outlier removal
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);       % smoothing

    %% Energy efficiency / energy rate trajectory
    % Using 2:end-1 to avoid edge artifacts/invalid endpoints in some datasets
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    %% Constant-current charge rate trajectory
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    %% Mid-point voltage trajectory
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    %% Platform discharge capacity trajectory
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    %% Plot trajectories (one figure per indicator)
    figure(1),hold on,box on,plot(Capa{Index,1},'-','color',ColorMap);
    figure(2),hold on,box on,plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3),hold on,box on,plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4),hold on,box on,plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5),hold on,box on,plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end