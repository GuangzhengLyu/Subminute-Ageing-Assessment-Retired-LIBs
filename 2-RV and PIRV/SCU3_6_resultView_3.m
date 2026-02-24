clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Conventional Full-Charge Relaxation Voltage vs Pulse-Inspection
%%%          Relaxation Voltage
%%% This script: Load SCU3 Dataset #3, build normalized target indicators,
%%% aggregate RMSE statistics from saved PLSR predictions across voltage
%%% setpoints, plot RMSE trends with error bars, and summarize relative RMSE
%%% improvements versus the full-charge case.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../OneCycle_3.mat')

%% Sample construction
% Filter valid samples and compute normalized health indicators used as
% regression targets (capacity, life, and expanded indicators).
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 1.75 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Normalized capacity proxy
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (normalized)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result aggregation
% Load saved prediction files for each voltage setpoint and compute RMSE
% across repeated runs.
for i = 1:14
    currentFile = sprintf('./Result/PLSR_Result_3_50_Y_Test_%d.mat',i);
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

% Voltage setpoints corresponding to the saved result index (1..14)
Uter = 3.0:0.1:4.3;

%% RMSE visualization and relative comparison
% For each target, plot mean RMSE with symmetric error bars and compute the
% normalized RMSE ratio versus the full-charge reference (index 14).
Temp = RMSE_Capa;
PM_RMSE(1,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(1),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(1,:) = mean(Temp,2);

Temp = RMSE_Life;
PM_RMSE(2,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(2),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(2,:) = mean(Temp,2);

Temp = RMSE_ERate;
PM_RMSE(3,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(3),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(3,:) = mean(Temp,2);

Temp = RMSE_CoChR;
PM_RMSE(4,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(4),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(4,:) = mean(Temp,2);

Temp = RMSE_MipV;
PM_RMSE(5,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(5),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(5,:) = mean(Temp,2); 

Temp = RMSE_PlatfCapa;
PM_RMSE(6,:) = mean(Temp,2)./mean(Temp(14,:),2);
figure(6),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(6,:) = mean(Temp,2);

% Summary bar plot: mean relative RMSE across all targets
figure,bar(mean(PM_RMSE,1)),grid on

% save M_RMSE_PLSR_3.mat M_RMSE