clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Load SCU3 Dataset #1, compute scaled health indicators, then
%%% read saved MLR leave-one-out prediction files across voltage setpoints,
%%% compute RMSE (over 100 repeats) for each task, and save mean RMSE curves.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset
load('../../OneCycle_1.mat')

%% Sample construction
% Filter samples by the ending step flag and extract life and scaled indicators
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

        % Scaled capacity-based SOH (relative to 3.5 Ah nominal reference)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (scaled by fixed references)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% RMSE evaluation across voltage setpoints
% For each setpoint file (i = 1..13) and each repetition (j = 1..100),
% compute task-wise RMSE between ground truth and predictions.
for i = 1:13
    currentFile = sprintf('MLR_Result_1_70_Y_Test_%d.mat',i);
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

%% Summary plotting and export
% Voltage terminal setpoints corresponding to the 13 indices
Uter = 3.0:0.1:4.2;

% Capacity-based SOH RMSE
Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,plot(Uter,mean(Temp,2),'-o')
M_RMSE(1,:) = mean(Temp,2);

% RUL RMSE
Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,plot(Uter,mean(Temp,2),'-s')
M_RMSE(2,:) = mean(Temp,2);

% Energy-efficiency-based indicator RMSE
Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,plot(Uter,mean(Temp,2),'-d')
M_RMSE(3,:) = mean(Temp,2);

% Constant-current charge-rate indicator RMSE
Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,plot(Uter,mean(Temp,2),'->')
M_RMSE(4,:) = mean(Temp,2);

% Mid-point voltage indicator RMSE
Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,plot(Uter,mean(Temp,2),'-^')
M_RMSE(5,:) = mean(Temp,2);

% Platform discharge capacity indicator RMSE
Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,plot(Uter,mean(Temp,2),'-<')
M_RMSE(6,:) = mean(Temp,2);

% Save mean RMSE curves (6 tasks × 13 setpoints)
save M_RMSE_MLR_1.mat M_RMSE