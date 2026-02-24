%%% ========================================================================
%%% Project : External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset : 7-MIT-A123
%%% This script: Visualize ageing trajectories (sorted by initial capacity)
%%%             with robust filtering (Hampel + Savitzky–Golay).
%%% ========================================================================

close all

% Total number of cells/samples
NumData = length(OneCycle);

% -------------------------------------------------------------------------
% Collect and sort samples by initial discharge capacity at CutCycle
% -------------------------------------------------------------------------
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A B] = sort(MyCapa);

for i = 1:NumData
    Index = B(i);

    % ---------------------------------------------------------------------
    % Colormap for early ageing visualization (rank-based)
    % ---------------------------------------------------------------------
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    % ---------------------------------------------------------------------
    % Alternative colormap for deep ageing visualization (life-based)
    % ---------------------------------------------------------------------
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1 0.4+0.6*(Life-190)/2000 0.2-0.2*(Life-190)/2000];

    % ---------------------------------------------------------------------
    % Smoothing settings
    % ---------------------------------------------------------------------
    FilterLength = 51;

    % ---------------------------------------------------------------------
    % Discharge capacity trajectory (robust filtering + smoothing)
    % ---------------------------------------------------------------------
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);

    % ---------------------------------------------------------------------
    % Energy efficiency trajectory (skip endpoints to avoid boundary artifacts)
    % ---------------------------------------------------------------------
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    % ---------------------------------------------------------------------
    % Constant-current charge ratio trajectory (skip endpoints)
    % ---------------------------------------------------------------------
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    % ---------------------------------------------------------------------
    % Mid-point voltage trajectory
    % ---------------------------------------------------------------------
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    % ---------------------------------------------------------------------
    % Plateau discharge capacity trajectory
    % ---------------------------------------------------------------------
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    % ---------------------------------------------------------------------
    % Plot: each cell is one curve, colored by rank-based ColorMap
    % ---------------------------------------------------------------------
    figure(1),hold on,box on,plot(Capa{Index,1},'-','color',ColorMap);
    figure(2),hold on,box on,plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3),hold on,box on,plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4),hold on,box on,plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5),hold on,box on,plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end