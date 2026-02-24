close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 9-SCU1-A123
%%% This script: Visualize ageing trajectories of multiple health indicators.
%%% Cells are sorted by initial capacity, then each indicator trajectory is
%%% filtered (Hampel outlier removal + Savitzky–Golay smoothing) and plotted
%%% with a capacity-ordered colormap to highlight early ageing trends.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    % Lower-capacity cells (more aged) appear "warmer" in this colormap
    ColorMap = [0.8*(NumData-i)/NumData, 0.5+0.8*(NumData-i)/(2*NumData), 0];

    %% Deep ageing colormap (alternative; disabled in the original code)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1, 0.4+0.6*(Life-190)/2000, 0.2-0.2*(Life-190)/2000];

    % Smoothing window length for Savitzky–Golay filtering (must be odd)
    FilterLength = 21;

    % Discharge capacity trajectory (Ah)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);                 % outlier removal
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);% smoothing

    % Energy efficiency / rate-like indicator (use middle segment only)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    % Constant-current charge rate indicator (use middle segment only)
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    % Mid-point voltage trajectory (V)
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    % Platform discharge capacity trajectory (Ah)
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    % Plot trajectories (each cell as one curve)
    figure(1), hold on, box on, plot(Capa{Index,1},'-','color',ColorMap);
    figure(2), hold on, box on, plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3), hold on, box on, plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4), hold on, box on, plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5), hold on, box on, plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end