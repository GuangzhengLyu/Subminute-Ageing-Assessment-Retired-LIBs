clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 10-Stanford-LMO
%%% This script: Visualize ageing trajectories (capacity and extended health
%%% indicators) across all cells. Cells are sorted by initial capacity, then
%%% each trajectory is denoised (Hampel + Savitzky-Golay) and plotted with a
%%% life-dependent colormap to highlight deep-ageing differences.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load("OneCycle_Stanford_LMO.mat")

% Total number of cells/samples
NumData = length(OneCycle);

% Collect initial (original) capacity of each sample for sorting
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort samples by initial capacity (ascending)
[A B] = sort(MyCapa);

for i = 1:NumData
    % Index of the i-th sample in sorted order
    Index = B(i);

    %% Early ageing (optional colormap design)
    % ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing (life-dependent colormap design)
    % Life is defined here as the length of the discharge-capacity trajectory
    Life = length(OneCycle(Index).Cycle.DiscCapaAh);

    % Colormap changes with life to emphasize deep ageing variation
    ColorMap = [1 0.4+0.6*(Life-1200)/2100 0.2-0.2*(Life-1200)/2100];

    % Smoothing window length for Savitzky-Golay filter
    FilterLength = 51;

    %% Discharge capacity trajectory (Ah)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);
    
    %% Energy efficiency / energy-rate indicator (dimensionless)
    % Note: use 2:end-1 to remove boundary points (as defined in the dataset)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    %% Constant-current charge rate indicator (dimensionless)
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    %% Mid-point (or minimum) voltage indicator (V)
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    %% Platform discharge capacity indicator (Ah)
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);
    
    % Plot all trajectories with consistent styling
    figure(1),hold on,box on,plot(Capa{Index,1},'-','color',ColorMap);
    figure(2),hold on,box on,plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3),hold on,box on,plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4),hold on,box on,plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5),hold on,box on,plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end