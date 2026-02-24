clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Post-process DNN predictions on SCU3 Dataset #2 (terminal-voltage sweep).
%%% It rebuilds six ground-truth targets (SOH, life/RUL proxy, and four expanded
%%% indicators).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
load('../../OneCycle_2.mat')

%% Build ground-truth targets (same definitions as model training)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData + 1;

        % Life: first cycle index where discharge capacity drops below 2.1 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Normalized SOH and expanded indicators (engineering normalization)
        Capa(CountData,1)      = OneCycle(IndexData).OrigCapaAh ./ 3.5;
        ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2)     / 89;
        CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2)  / 83;
        MindVolt(CountData,1)  = (OneCycle(IndexData).Cycle.MindVoltV(1) - 2.65) / (3.47 - 2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)     / 1.3;
    end
end

%% Load DNN results and compute RMSE across repeats and setpoints
for i = 1:13
    currentFile = sprintf('DNN_Result_2_60_Y_Test_%d.mat', i);
    load(currentFile)  % loads Y_Test: (repeat, task, sample)

    for j = 1:100
        % Each Y_Test(j,task,:) is a vector of predictions over all samples
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Summarize RMSE vs terminal voltage with error bars (mean ± 1 sigma)
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

%% Save mean RMSE curves (6 tasks × 13 setpoints)
save M_RMSE_DNN_2.mat M_RMSE