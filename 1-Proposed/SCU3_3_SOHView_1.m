clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Build a filtered sample set from SCU3 Dataset #1, compute
%%% expanded health indicators in capacity bins, normalize each indicator,
%%% and plot the mean normalized indicator versus initial capacity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset
load('../OneCycle_1.mat')

%% Sample construction
% Filter samples by the ending step flag and extract life, capacity, and
% expanded health indicators at the specified cycle indices.
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 2.5 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity (used for binning)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (cycle-index selection follows raw data structure)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Expanded health indicators (binned by original capacity)
% Group samples into 0.1 Ah bins starting from 2.8 Ah, and collect indicator
% values within each bin for subsequent normalization and statistics.
InCaCo = 0;
for InCa = 2.8:0.1:3.4
    InCaCo = InCaCo+1;

    % Identify samples with original capacity in (InCa, InCa+0.1)
    A = find(InCa<Capa);
    B = find(Capa<InCa+0.1);

    % Collect indicator values within the capacity bin
    ERCa{InCaCo} = ERate(intersect(A,B));
    CCRCa{InCaCo} = CoChRate(intersect(A,B));
    MVCa{InCaCo} = MindVolt(intersect(A,B));
    PCCa{InCaCo} = PlatfCapa(intersect(A,B));
end

% Capacity bin centers/edges used for plotting
InCa = 2.8:0.1:3.4;

%% Normalization and statistics: EnergyRate
% Normalize by the reference scale (89), then compute mean/max/min/std per bin
Temp = ERCa;

for i =1:7
    Temp{i} = Temp{i}/89;
    MeanTemp(i) = mean(Temp{i});
    MaxTemp(i) = max(Temp{i});
    MinTemp(i) = min(Temp{i});
    StdTemp(i) = std(Temp{i});
end

% Plot mean normalized indicator versus capacity bins
figure(1),hold on,plot(InCa,MeanTemp,'d-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: ConstCharRate
% Normalize by the reference scale (83), then compute mean/max/min/std per bin
Temp = CCRCa;

for i =1:7
    Temp{i} = Temp{i}/83;
    MeanTemp(i) = mean(Temp{i});
    MaxTemp(i) = max(Temp{i});
    MinTemp(i) = min(Temp{i});
    StdTemp(i) = std(Temp{i});
end

figure(1),hold on,plot(InCa,MeanTemp,'>-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: MindVoltV
% Min-max normalization using fixed bounds (2.65, 3.47)
Temp = MVCa;
for i =1:7
    Temp{i} = (Temp{i}-2.65)/(3.47-2.65);
    MeanTemp(i) = mean(Temp{i});
    MaxTemp(i) = max(Temp{i});
    MinTemp(i) = min(Temp{i});
    StdTemp(i) = std(Temp{i});
end

figure(1),hold on,plot(InCa,MeanTemp,'^-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: PlatfCapaAh
% Normalize by the reference scale (1.3), then compute mean/max/min/std per bin
Temp = PCCa;
for i =1:7
    Temp{i} = Temp{i}/1.3;
    MeanTemp(i) = mean(Temp{i});
    MaxTemp(i) = max(Temp{i});
    MinTemp(i) = min(Temp{i});
    StdTemp(i) = std(Temp{i});
end

figure(1),hold on,plot(InCa,MeanTemp,'<-'),axis([1.9,3.5,0,1.1])