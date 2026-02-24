clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Build a filtered sample set from SCU3 Dataset #2, extract
%%% expanded health indicators, normalize each indicator using fixed scales,
%%% and plot the mean normalized indicator at a single capacity point.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load('../OneCycle_2.mat')

%% Sample construction
% Filter samples by the ending step flag and extract life, capacity, and
% expanded health indicators at the specified cycle indices.
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 2.1 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity (stored but not used for binning in this script)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (cycle-index selection follows raw data structure)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Expanded health indicators (single-bin aggregation)
% Dataset #2 is treated as a single capacity bin in this script. All samples
% contribute to the same group.
InCaCo = 1;

ERCa{InCaCo} = ERate;
CCRCa{InCaCo} = CoChRate;
MVCa{InCaCo} = MindVolt;
PCCa{InCaCo} = PlatfCapa;

% Single x-location used for plotting (representative capacity point)
InCa = 2.5;

%% Normalization and statistics: EnergyRate
% Normalize by the reference scale (89), then compute mean/max/min/std
Temp = ERCa;

i = 1;
Temp{i} = Temp{i}/89;
MeanTemp(i) = mean(Temp{i});
MaxTemp(i) = max(Temp{i});
MinTemp(i) = min(Temp{i});
StdTemp(i) = std(Temp{i});

figure(1),hold on,plot(InCa,MeanTemp,'d-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: ConstCharRate
% Normalize by the reference scale (83), then compute mean/max/min/std
Temp = CCRCa;

i = 1;
Temp{i} = Temp{i}/83;
MeanTemp(i) = mean(Temp{i});
MaxTemp(i) = max(Temp{i});
MinTemp(i) = min(Temp{i});
StdTemp(i) = std(Temp{i});

figure(1),hold on,plot(InCa,MeanTemp,'>-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: MindVoltV
% Min-max normalization using fixed bounds (2.65, 3.47)
Temp = MVCa;
i = 1;
Temp{i} = (Temp{i}-2.65)/(3.47-2.65);
MeanTemp(i) = mean(Temp{i});
MaxTemp(i) = max(Temp{i});
MinTemp(i) = min(Temp{i});
StdTemp(i) = std(Temp{i});

figure(1),hold on,plot(InCa,MeanTemp,'^-'),axis([1.9,3.5,0,1.1])

%% Normalization and statistics: PlatfCapaAh
% Normalize by the reference scale (1.3), then compute mean/max/min/std
Temp = PCCa;

i = 1;
Temp{i} = Temp{i}/1.3;
MeanTemp(i) = mean(Temp{i});
MaxTemp(i) = max(Temp{i});
MinTemp(i) = min(Temp{i});
StdTemp(i) = std(Temp{i});

figure(1),hold on,plot(InCa,MeanTemp,'<-'),axis([1.9,3.5,0,1.1])