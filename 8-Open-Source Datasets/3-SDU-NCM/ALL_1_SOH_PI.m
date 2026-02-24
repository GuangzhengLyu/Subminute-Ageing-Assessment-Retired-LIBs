close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 3-SDU-NCM
%%% This script: Visualize ageing trajectories of multiple cells by plotting
%%% smoothed time-series curves of key indicators (discharge capacity, energy
%%% efficiency, constant-current charge rate, mid-point voltage, and platform
%%% discharge capacity). Cells are sorted by original capacity, and curve
%%% colors are assigned as a function of cycle-life length (deep-ageing view).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of samples/cells in the dataset
NumData = length(OneCycle);

%% Sort cells by original capacity (ascending)
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

    % Smoothing configuration (kept as original)
    FilterLength = 51;

    %% Discharge capacity trajectory (Ah)
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);

    %% Energy-rate trajectory (use valid interior range as in original)
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    %% Constant-current charge-rate trajectory
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    %% Mid-point voltage trajectory (V)
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    %% Platform discharge capacity trajectory (Ah)
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);

    %% Plot trajectories (each cell as one curve)
    figure(1),hold on,box on,plot(Capa{Index,1},'-','color',ColorMap);
    figure(2),hold on,box on,plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3),hold on,box on,plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4),hold on,box on,plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5),hold on,box on,plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end