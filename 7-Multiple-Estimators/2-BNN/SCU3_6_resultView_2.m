clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Post-process Bayesian neural network (BNN, trainbr) results
%%% for SCU3 Dataset #2 by loading saved Y_Test files, computing per-task RMSE
%%% across repetitions and voltage setpoints, plotting mean RMSE curves with
%%% error bars (std across repetitions), and saving the aggregated RMSE
%%% matrix (M_RMSE).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load('../../OneCycle_2.mat')

%% Sample construction (ground truth)
% Filter samples by the ending step flag and reconstruct normalized outputs:
% capacity-based SOH, life (cycle index), and expanded health indicators.
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
        % Normalized capacity-based SOH proxy
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (scaled to unified ranges)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result aggregation (RMSE over repetitions and voltage setpoints)
% For each voltage setpoint file (i = 1..13), load Y_Test and compute RMSE
% for each task over 100 repeated runs.
for i = 1:13
    currentFile = sprintf('BNN_Result_2_60_Y_Test_%d.mat',i);
    load(currentFile)
    for j = 1:100
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Visualization and summary statistics
% Uter defines the voltage setpoint axis corresponding to i = 1..13.
Uter = 3.0:0.1:4.2;

% Task 1: Capacity-based SOH proxy
Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(1,:) = mean(Temp,2);

% Task 2: Life (cycle index)
Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(2,:) = mean(Temp,2);

% Task 3: Energy efficiency indicator
Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(3,:) = mean(Temp,2);

% Task 4: Constant-current charge-rate indicator
Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(4,:) = mean(Temp,2);

% Task 5: Mid-point voltage indicator
Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(5,:) = mean(Temp,2);

% Task 6: Platform discharge capacity indicator
Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(6,:) = mean(Temp,2);

%% Save aggregated RMSE matrix
% M_RMSE has size [6 x 13]: rows correspond to tasks, columns to voltage setpoints.
save M_RMSE_BNN_2.mat M_RMSE