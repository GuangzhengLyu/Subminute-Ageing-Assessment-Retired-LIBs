close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% This script: Visualize ageing trajectories across all cells by sorting
%%% samples using initial discharge capacity (OrigCapaAh), then plotting
%%% smoothed per-cycle trajectories of capacity and expanded performance
%%% indicators. Outliers are suppressed using Hampel filtering, and trends
%%% are smoothed using Savitzky–Golay filtering. Line colors encode ageing
%%% severity (default: early-ageing colormap; deep-ageing option provided).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Sort cells by initial discharge capacity (OrigCapaAh)
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A, B] = sort(MyCapa); %#ok<ASGLU>

%% Plot smoothed trajectories for each cell (sorted order)
for i = 1:NumData
    Index = B(i);

    %% Early-ageing color mapping (higher initial capacity -> darker tone)
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep-ageing color mapping option (based on cycle life)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    %% Filtering settings
    FilterLength = 21;   % Window length for Savitzky–Golay filter (must be odd)

    %% Discharge capacity trajectory (Ah)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);

    %% Energy efficiency trajectory (dimensionless)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    %% Constant-current charge ratio trajectory (dimensionless)
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    %% Mid-point voltage trajectory (V)
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    %% Voltage-plateau discharge capacity trajectory (Ah)
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    %% Plot trajectories (each figure overlays all cells)
    figure(1), hold on, box on, plot(Capa{Index,1},'-','color',ColorMap);
    figure(2), hold on, box on, plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3), hold on, box on, plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4), hold on, box on, plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5), hold on, box on, plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end