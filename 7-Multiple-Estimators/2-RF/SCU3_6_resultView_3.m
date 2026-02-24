clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Post-process Random Forest (RF) results for SCU3 Dataset #3.
%%% It reconstructs normalized ground-truth outputs, loads saved RF predictions
%%% across voltage setpoints, computes per-task RMSE over repetitions, and
%%% visualizes RMSE mean ± std versus terminal-voltage setpoint.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../../OneCycle_3.mat')

%% Sample construction and label reconstruction
% Filter samples by the ending step flag and extract life, normalized capacity,
% and expanded health indicators (all in the same normalized scales used by RF).
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

        % Capacity-based SOH (normalized by nominal capacity)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (normalized to reference ranges)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% RMSE evaluation across voltage setpoints and repetitions
% Load RF prediction files for each setpoint (i = 1..13) and compute RMSE
% for each output dimension over repeated runs (j = 1..100).
for i = 1:13
    currentFile = sprintf('RF_Result_3_50_Y_Test_%d.mat',i);
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

%% RMSE visualization versus terminal-voltage setpoint
% Uter corresponds to the terminal-voltage setpoints (3.0 V to 4.2 V, step 0.1 V).
Uter = 3.0:0.1:4.2;

% 1) Capacity-based SOH RMSE
Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(1,:) = mean(Temp,2);

% 2) RUL (life) RMSE
Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(2,:) = mean(Temp,2);

% 3) Energy efficiency RMSE
Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(3,:) = mean(Temp,2);

% 4) Constant-current charge-rate RMSE
Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(4,:) = mean(Temp,2);

% 5) Mid-point voltage RMSE
Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(5,:) = mean(Temp,2); 

% 6) Platform discharge capacity RMSE
Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,errorbar(Uter,mean(Temp,2),std(Temp',1),std(Temp',1),'-')
M_RMSE(6,:) = mean(Temp,2);

%% Save summary results
% M_RMSE stores the mean RMSE (over repetitions) for each task and setpoint.
save M_RMSE_RF_3.mat M_RMSE