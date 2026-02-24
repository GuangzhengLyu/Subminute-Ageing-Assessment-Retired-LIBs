close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Visualize per-cell ageing trajectories of capacity and
%%% multiple expanded performance indicators. Each cell is color-coded by
%%% its initial/original capacity after sorting, and signals are denoised
%%% using Hampel filtering and Savitzky–Golay smoothing for clearer trends.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of cells/samples
NumData = length(OneCycle);

%% Sort samples by original capacity (ascending)
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A B] = sort(MyCapa);

%% Plot denoised trajectories for each sample (color-coded by sorted index)
for i = 1:NumData
    Index = B(i);

    %% Early ageing color map (capacity-ranked)
    % Color gradually changes with the sorted index to visualize heterogeneity
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing color map (optional; keep the original commented logic)
    % if min(OneCycle(Index).Cycle.DiscCapaAh)<=2.1
    %     Life = min(find(OneCycle(Index).Cycle.DiscCapaAh <= 2.1));
    % else
    %     Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % end
    % ColorMap = [1 0.4+0.6*(Life-200)/605 0.2-0.2*(Life-200)/605];

    % Smoothing window length for sgolayfilt (must be odd)
    FilterLength = 51;

    %% Discharge capacity (Ah)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);

    %% Energy efficiency proxy (EnergyRate)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    %% Constant-current charge rate proxy (ConstCharRate)
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    %% Mid-point voltage (MindVoltV)
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    %% Platform discharge capacity (PlatfCapaAh)
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