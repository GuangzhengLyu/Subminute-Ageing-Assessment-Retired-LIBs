clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Post-process CNN results on SCU3 Dataset #1.
%%% It reconstructs six ground-truth targets (SOH, RUL proxy, and four
%%% expanded indicators), loads CNN predictions saved per terminal-voltage
%%% setpoint, computes RMSE across 100 repeated runs, visualizes mean ± std
%%% RMSE versus terminal voltage, and saves the mean RMSE curves (M_RMSE).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
load('../../OneCycle_1.mat')

%% Build ground-truth targets for evaluation
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

        % Capacity-based SOH (normalized by nominal reference capacity)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (normalized by dataset-specific references)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Load CNN predictions and compute RMSE for each setpoint and repeat
for i = 1:13
    currentFile = sprintf('CNN_Result_1_70_Y_Test_%d.mat',i);
    load(currentFile) % expects Y_Test: [repeat, task(1..6), sample]

    for j = 1:100
        % Task 1: Capacity-based SOH
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));

        % Task 2: Life / RUL proxy (cycle count)
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));

        % Tasks 3–6: Expanded indicators
        RMSE_ERate(i,j)      = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)      = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)       = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j)  = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Summarize RMSE versus terminal voltage and visualize mean ± std
Uter = 3.0:0.1:4.2;

Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(1,:) = mean(Temp,2);

Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(2,:) = mean(Temp,2);

Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(3,:) = mean(Temp,2);

Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(4,:) = mean(Temp,2);

Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(5,:) = mean(Temp,2);

Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on
errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(6,:) = mean(Temp,2);

%% Save mean RMSE curves for downstream plotting/aggregation
save M_RMSE_CNN_1.mat M_RMSE