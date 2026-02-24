clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Post-process PLSR leave-one-out results -> compute RMSE curves
%%% across voltage setpoints on SCU3 Dataset #1 (multi-task ageing indicators).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset
load('../../OneCycle_1.mat')

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (pre-normalized to unified scales)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result aggregation (load saved predictions and compute RMSE)
% Each file contains Y_Test for one setpoint index i (1..13), and each Y_Test(j,:,:)
% corresponds to the j-th repeat in the repeated leave-one-out loop.
for i = 1:13
    currentFile = sprintf('./PLSR_Result_1_70_Y_Test_%d.mat',i);
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

%% Example scatter plots and error vectors (uses the last loaded Y_Test and last j)
figure(7),hold on,plot(Capa,squeeze(Y_Test(j,1,:)),'o')
ErCA_1 = Capa-squeeze(Y_Test(j,1,:));
figure(8),hold on,plot(Life,squeeze(Y_Test(j,2,:)),'s')
ErLI_1 = Life-squeeze(Y_Test(j,2,:));
figure(9),hold on,plot(ERate,squeeze(Y_Test(j,3,:)),'>')
ErER_1 = ERate-squeeze(Y_Test(j,3,:));
figure(10),hold on,plot(CoChRate,squeeze(Y_Test(j,4,:)),'d')
ErCR_1 = CoChRate-squeeze(Y_Test(j,4,:));
figure(11),hold on,plot(MindVolt,squeeze(Y_Test(j,5,:)),'^')
ErMV_1 = MindVolt-squeeze(Y_Test(j,5,:));
figure(12),hold on,plot(PlatfCapa,squeeze(Y_Test(j,6,:)),'>')
ErPC_1 = PlatfCapa-squeeze(Y_Test(j,6,:)); 
  
% Voltage terminal grid (3.0 V to 4.2 V with 0.1 V step; 13 setpoints)
Uter = 3.0:0.1:4.2;

%% Normalized RMSE aggregation (task-wise scaling and averaged score)
Temp(1,:) = mean(RMSE_Capa,2)/0.0446;
Temp(2,:) = mean(RMSE_Life,2)/66.6030;
Temp(3,:) = mean(RMSE_ERate,2)/0.0176;
Temp(4,:) = mean(RMSE_CoChR,2)/0.0286;
Temp(5,:) = mean(RMSE_MipV,2)/0.0418;
Temp(6,:) = mean(RMSE_PlatfCapa,2)/0.1068;

MyTemp = mean(Temp,1);

%% Summary plot (average normalized RMSE vs setpoint index)
figure(1),hold on,plot(MyTemp,'-o'),box on,grid on