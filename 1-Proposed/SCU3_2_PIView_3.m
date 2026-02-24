clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Sort Dataset #3 cells by original capacity and visualize
%%% multi-dimensional ageing indicators with color mapping by original capacity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dataset #3
% Load structured single-cycle dataset
load('../OneCycle_3.mat')

% Total number of samples in Dataset #3
NumData = 433;

% Extract original discharge capacity for sorting
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells in ascending order of original capacity
[A B] = sort(MyCapa);

% Traverse cells according to sorted order
for i = 1:NumData

    % Recover original index after sorting
    Index = B(i);

    %% Dataset #3 (indicator extraction and visualization)

    % Define color mapping according to original capacity level
    ColorMap = [1 (OneCycle(Index).OrigCapaAh-2.0)/0.5 0];

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
    axis([0,2000,1.75,2.5])

    % Plot energy efficiency 
    figure(2),hold on,plot(EnergyRate{Index,1},'.','color',ColorMap);
    axis([0,2000,62,80])

    % Plot constant-current charge rate 
    figure(3),hold on,plot(ConstCharRate{Index,1},'.','color',ColorMap);
    axis([0,2000,0,70])

    % Plot mid-point voltage 
    figure(4),hold on,plot(MindVoltV{Index,1},'.','color',ColorMap);
    axis([0,2000,2.65,3.3])

    % Plot platform discharge capacity 
    figure(5),hold on,plot(PlatfCapaAh{Index,1},'.','color',ColorMap);
    axis([0,2000,0,0.5])

end