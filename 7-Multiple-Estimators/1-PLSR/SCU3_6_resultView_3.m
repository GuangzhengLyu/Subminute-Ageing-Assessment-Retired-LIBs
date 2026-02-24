clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Compute RMSE curves of repeated-test PLSR predictions for
%%% SCU3 Dataset #3 across voltage setpoints, then save the mean RMSE traces.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../../OneCycle_3.mat')

%% Sample construction
% Filter samples by the ending step flag and extract life and expanded
% health indicators for RMSE evaluation (scaled to unified indicator forms).
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData + 1;

        % Define life as the first cycle where discharge capacity drops below 1.75 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Unified health-indicator scaling (consistent with model output units)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% RMSE evaluation across voltage setpoints
% For each setpoint file, compute RMSE over 100 repetitions for all tasks.
for i = 1:13
    currentFile = sprintf('./5/PLSR_Result_3_50_Y_Test_%d.mat', i);
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

%% RMSE curve visualization and aggregation
% Uter corresponds to the 13 voltage setpoints used in feature extraction.
Uter = 3.0:0.1:4.2;

Temp = RMSE_Capa;
[1, mean(Temp(13,:)), std(Temp(13,:))]
figure(1), hold on, plot(Uter, mean(Temp,2), '-o')
M_RMSE(1,:) = mean(Temp,2);

Temp = RMSE_Life;
[2, mean(Temp(13,:)), std(Temp(13,:))]
figure(2), hold on, plot(Uter, mean(Temp,2), '-s')
M_RMSE(2,:) = mean(Temp,2);

Temp = RMSE_ERate;
[3, mean(Temp(13,:)), std(Temp(13,:))]
figure(3), hold on, plot(Uter, mean(Temp,2), '-d')
M_RMSE(3,:) = mean(Temp,2);

Temp = RMSE_CoChR;
[4, mean(Temp(13,:)), std(Temp(13,:))]
figure(4), hold on, plot(Uter, mean(Temp,2), '->')
M_RMSE(4,:) = mean(Temp,2);

Temp = RMSE_MipV;
[5, mean(Temp(13,:)), std(Temp(13,:))]
figure(5), hold on, plot(Uter, mean(Temp,2), '-^')
M_RMSE(5,:) = mean(Temp,2);

Temp = RMSE_PlatfCapa;
[6, mean(Temp(13,:)), std(Temp(13,:))]
figure(6), hold on, plot(Uter, mean(Temp,2), '-<')
M_RMSE(6,:) = mean(Temp,2);

%% Save aggregated RMSE traces
save M_RMSE_PLSR_3.mat M_RMSE