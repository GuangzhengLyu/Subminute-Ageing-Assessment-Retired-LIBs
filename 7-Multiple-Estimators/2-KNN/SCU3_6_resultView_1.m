clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Summarize KNN results on SCU3 Dataset #1 (updated file: OneCycle_1_260102.mat).
%%% It reconstructs six scaled health indicators (ground truth), loads saved
%%% KNN predictions (Y_Test) across 13 relaxation-voltage setpoints and 100
%%% repeats, computes RMSE for each task, visualizes mean±std RMSE versus
%%% terminal voltage, and saves the mean RMSE curves (M_RMSE).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
load('../OneCycle_1_260102.mat')

%% Build ground-truth targets (scaled health indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Life definition: first cycle where discharge capacity drops below 2.5 Ah
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH (scaled)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (scaled)
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1)  = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Load predictions and compute RMSE (13 setpoints × 100 repeats)
for i = 1:13
    currentFile = sprintf('KNN_Result_1_70_Y_Test_%d.mat', i);
    load(currentFile)  % loads Y_Test: (repeat, task, sample)

    for j = 1:100
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Visualize mean±std RMSE versus terminal voltage and save mean curves
Uter = 3.0:0.1:4.2;

Temp = RMSE_Capa;
[1, mean(Temp(13,:)), std(Temp(13,:))]
figure(1), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(1,:) = mean(Temp,2);

Temp = RMSE_Life;
[2, mean(Temp(13,:)), std(Temp(13,:))]
figure(2), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(2,:) = mean(Temp,2);

Temp = RMSE_ERate;
[3, mean(Temp(13,:)), std(Temp(13,:))]
figure(3), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(3,:) = mean(Temp,2);

Temp = RMSE_CoChR;
[4, mean(Temp(13,:)), std(Temp(13,:))]
figure(4), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(4,:) = mean(Temp,2);

Temp = RMSE_MipV;
[5, mean(Temp(13,:)), std(Temp(13,:))]
figure(5), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(5,:) = mean(Temp,2);

Temp = RMSE_PlatfCapa;
[6, mean(Temp(13,:)), std(Temp(13,:))]
figure(6), hold on
errorbar(Uter, mean(Temp,2), std(Temp',1), std(Temp',1), '-')
M_RMSE(6,:) = mean(Temp,2);

save M_RMSE_KNN_1.mat M_RMSE