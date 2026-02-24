clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Sort Dataset #1 cells by original capacity and visualize
%%% multiple ageing-related indicators (capacity and expanded features)
%%% along cycling trajectories with color mapping by capacity ranking.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dataset #1
% Load single-cycle structured data
load('../OneCycle_1.mat')

% Total number of samples considered in Dataset #1
NumData = 105;

% Extract original discharge capacity (used for ranking)
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells in ascending order of original capacity
[A B] = sort(MyCapa);

% Traverse cells in sorted order (low to high capacity)
for i = 1:NumData

    % Get original index after sorting
    Index = B(i);

    %% Dataset #1 (indicator extraction and visualization)
    % Define color gradient according to capacity ranking
    % Cells with lower capacity appear darker in the colormap
    ColorMap = [0.8*(NumData-i)/NumData ...
                0.5+0.8*(NumData-i)/(2*NumData) ...
                0];

    % Discharge capacity trajectory
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;

    % Energy efficiency trajectory (excluding boundary points)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    ER(i) = EnergyRate{Index,1}(1);

    % Constant-current charge rate trajectory
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    CCR(i) = ConstCharRate{Index,1}(1);

    % Mid-point voltage trajectory
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MV(i) = MindVoltV{Index,1}(1);

    % Platform discharge capacity trajectory
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PC(Index) = PlatfCapaAh{Index,1}(1);

    % Plot discharge capacity 
    figure(1),hold on,plot(Capa{Index,1},'.','color',ColorMap);
    axis([0,450,2.5,3.5])

    % Plot energy efficiency 
    figure(2),hold on,plot(EnergyRate{Index,1},'.','color',ColorMap);
    axis([0,450,73,91])

    % Plot constant-current charge rate 
    figure(3),hold on,plot(ConstCharRate{Index,1},'.','color',ColorMap);
    axis([0,450,45,85])

    % Plot mid-point voltage 
    figure(4),hold on,plot(MindVoltV{Index,1},'.','color',ColorMap);
    axis([0,450,2.95,3.5])

    % Plot platform discharge capacity
    figure(5),hold on,plot(PlatfCapaAh{Index,1},'.','color',ColorMap);
    axis([0,450,0,1.4])

end