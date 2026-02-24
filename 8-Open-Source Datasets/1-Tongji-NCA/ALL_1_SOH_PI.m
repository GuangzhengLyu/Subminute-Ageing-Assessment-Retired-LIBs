close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% Description:
%%%   Sort cells by BOL discharge capacity (OrigCapaAh) and visualize the
%%%   cycle-resolved ageing trajectories for multiple indicators.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NumData = length(OneCycle);

% Collect BOL capacity of each cell for sorting
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells by BOL capacity (ascending)
[A B] = sort(MyCapa);

for i = 1:NumData
    Index = B(i);

    %% Early ageing (colormap based on sorted BOL capacity rank)
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing (alternative colormap based on cycle life; disabled by default)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1 0.4+0.6*(Life-190)/2000 0.2-0.2*(Life-190)/2000];

    % Smoothing window length for Savitzky–Golay filtering (must be odd)
    FilterLength = 51;

    % Discharge capacity trajectory (Ah): outlier removal + smoothing
    Capa{Index,1} = OneCycle(Index).Cycle.DiscCapaAh;
    Capa{Index,1} = hampel(Capa{Index,1},10);
    Capa{Index,1} = sgolayfilt(Capa{Index,1}, 3, FilterLength);
    
    % Energy efficiency trajectory: use interior points to avoid edge effects
    EnergyRate{Index,1} = OneCycle(Index).Cycle.EnergyRate(2:end-1);
    EnergyRate{Index,1} = hampel(EnergyRate{Index,1},100);
    EnergyRate{Index,1} = sgolayfilt(EnergyRate{Index,1}, 3, FilterLength);

    % Constant-current charge ratio trajectory: use interior points to avoid edge effects
    ConstCharRate{Index,1} = OneCycle(Index).Cycle.ConstCharRate(2:end-1);
    ConstCharRate{Index,1} = hampel(ConstCharRate{Index,1},20);
    ConstCharRate{Index,1} = sgolayfilt(ConstCharRate{Index,1}, 3, FilterLength);

    % Mid-point voltage trajectory (V): outlier removal + smoothing
    MindVoltV{Index,1} = OneCycle(Index).Cycle.MindVoltV;
    MindVoltV{Index,1} = hampel(MindVoltV{Index,1},100);
    MindVoltV{Index,1} = sgolayfilt(MindVoltV{Index,1}, 3, FilterLength);

    % Discharge plateau capacity trajectory (Ah): outlier removal + smoothing
    PlatfCapaAh{Index,1} = OneCycle(Index).Cycle.PlatfCapaAh;
    PlatfCapaAh{Index,1} = hampel(PlatfCapaAh{Index,1},10);
    PlatfCapaAh{Index,1} = sgolayfilt(PlatfCapaAh{Index,1}, 3, FilterLength);
    
    % Plot trajectories (one figure per indicator); color encodes cell ordering
    figure(1),hold on,box on,plot(Capa{Index,1},'-','color',ColorMap);
    figure(2),hold on,box on,plot(EnergyRate{Index,1},'-','color',ColorMap);
    figure(3),hold on,box on,plot(ConstCharRate{Index,1},'-','color',ColorMap);
    figure(4),hold on,box on,plot(MindVoltV{Index,1},'-','color',ColorMap);
    figure(5),hold on,box on,plot(PlatfCapaAh{Index,1},'-','color',ColorMap);
end