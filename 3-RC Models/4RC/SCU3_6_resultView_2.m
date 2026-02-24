clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Evaluate 4RC-based PLSR predictions on SCU3 Dataset #2
%%% - Build targets: SOH (capacity-based), RUL proxy, and four expanded indicators
%%% - Load saved PLSR predictions (Y_Test) for each relaxation-voltage setpoint
%%% - Compute RMSE over 100 repeated runs for each setpoint (13 setpoints)
%%% - Aggregate a normalized multi-task score (MyTemp) across 6 targets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
load('../../OneCycle_2.mat')

%% Sample construction
% Select valid samples and construct normalized targets:
% Capa: normalized by 3.5 Ah
% ERate: normalized by 89
% CoChRate: normalized by 83
% MindVolt: normalized by (3.47-2.65) with offset 2.65
% PlatfCapa: normalized by 1.3 Ah
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % RUL proxy: first cycle index where discharge capacity drops below threshold (2.1 Ah)
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH (normalized)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (normalized)
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1)  = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result evaluation
% For each setpoint index i = 1..13:
% - Load prediction file containing Y_Test(CountRP, task, sample)
% - Compute RMSE across samples for each of 100 repeated runs
for i = 1:13
    currentFile = sprintf('./PLSR_Result_2_60_Y_Test_%d_4RC.mat',i);
    load(currentFile)

    for j = 1:100
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Example scatter plots (uses the last loaded Y_Test and last j from the loops)
% Note: j retains its final value (100) after the loop above.
figure(7), hold on, plot(Capa,      squeeze(Y_Test(j,1,:)), 'o')
ErCA_1 = Capa - squeeze(Y_Test(j,1,:));

figure(8), hold on, plot(Life,      squeeze(Y_Test(j,2,:)), 's')
ErLI_1 = Life - squeeze(Y_Test(j,2,:));

figure(9), hold on, plot(ERate,     squeeze(Y_Test(j,3,:)), '>')
ErER_1 = ERate - squeeze(Y_Test(j,3,:));

figure(10), hold on, plot(CoChRate, squeeze(Y_Test(j,4,:)), 'd')
ErCR_1 = CoChRate - squeeze(Y_Test(j,4,:));

figure(11), hold on, plot(MindVolt, squeeze(Y_Test(j,5,:)), '^')
ErMV_1 = MindVolt - squeeze(Y_Test(j,5,:));

figure(12), hold on, plot(PlatfCapa, squeeze(Y_Test(j,6,:)), '>')
ErPC_1 = PlatfCapa - squeeze(Y_Test(j,6,:));

% Voltage setpoints (not directly used below, but kept for reference)
Uter = 3.0:0.1:4.2;

%% Aggregate normalized multi-task score across setpoints
% Each row corresponds to one task; values are averaged over 100 repeats.
% Denominators are task-specific scaling constants used for normalization.
Temp(1,:) = mean(RMSE_Capa,2)/0.0020;
Temp(2,:) = mean(RMSE_Life,2)/157.7464;
Temp(3,:) = mean(RMSE_ERate,2)/0.0161;
Temp(4,:) = mean(RMSE_CoChR,2)/0.0613;
Temp(5,:) = mean(RMSE_MipV,2)/0.0770;
Temp(6,:) = mean(RMSE_PlatfCapa,2)/0.0689;

% Average across the six tasks to obtain a single score per setpoint index
MyTemp = mean(Temp,1);

% Plot multi-task score versus setpoint index (1..13)
figure(1), hold on, plot(MyTemp,'-o'), box on, grid on