close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Visualize per-cell ageing trajectories with smoothing and
%%% outlier suppression, using a color gradient based on initial capacity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumData = length(OneCycle);

% Collect initial (reference) capacity for sorting / ranking
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells by initial capacity (ascending)
[A, B] = sort(MyCapa); %#ok<ASGLU>

for i = 1:NumData
    Index = B(i);

    % =========================
    % Color mapping for visualization
    % - Early ageing: color gradient driven by capacity rank
    % =========================
    ColorMap = [ ...
        0.8*(NumData-i)/NumData, ...
        0.5 + 0.8*(NumData-i)/(2*NumData), ...
        0 ...
        ];

    % =========================
    % Alternative: color mapping for deep ageing (disabled)
    % Life can be defined by reaching a discharge-capacity threshold.
    % =========================
    % if min(OneCycle(Index).Cycle.DiscCapaAh)<=2.1
    %     Life = min(find(OneCycle(Index).Cycle.DiscCapaAh <= 2.1));
    % else
    %     Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % end
    % ColorMap = [1 0.4+0.6*(Life-200)/605 0.2-0.2*(Life-200)/605];

    % =========================
    % Smoothing parameters
    % - Savitzky-Golay: polynomial order 3, window length 51
    % - Note: FilterLength should be odd and smaller than the data length.
    % =========================
    FilterLength = 51;

    % =========================
    % 1) Discharge capacity trajectory
    % =========================
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);

    % =========================
    % 2) Energy efficiency trajectory
    % Note: exclude the first and last points to avoid boundary artifacts
    % =========================
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    % =========================
    % 3) Constant-current charge fraction trajectory
    % Note: exclude the first and last points to avoid boundary artifacts
    % =========================
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    % =========================
    % 4) Mid-point voltage trajectory
    % =========================
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    % =========================
    % 5) Discharge plateau capacity trajectory
    % =========================
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    % =========================
    % Plot all trajectories with the same cell-specific color
    % =========================
    figure(1), hold on, box on, plot(Capa{Index,1},'-','color',ColorMap);
    figure(2), hold on, box on, plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3), hold on, box on, plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4), hold on, box on, plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5), hold on, box on, plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end