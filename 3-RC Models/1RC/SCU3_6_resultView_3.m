clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Evaluate 1-RC PLSR results on SCU3 Dataset #3 by computing
%%% multi-task RMSE across voltage setpoints and visualizing representative
%%% prediction scatter plots plus an aggregated normalized RMSE profile.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../../OneCycle_3.mat')

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;
        % Expanded health indicators
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result evaluation (RMSE)
% Load saved predictions (Y_Test) for each voltage setpoint index i = 1..13,
% then compute RMSE over 100 repeated runs for six ageing assessment tasks.
for i = 1:13
    currentFile = sprintf('./PLSR_Result_3_50_Y_Test_%d_1RC.mat',i);
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

%% Scatter plots for a representative run (j from the last loop iteration)
% These figures compare ground truth vs. predicted values and store residuals.
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
  
%% Aggregated normalized RMSE profile
% Normalize the mean RMSE of each task by a fixed reference scalar, then
% average across tasks to obtain an overall performance curve vs. setpoint.
Uter = 3.0:0.1:4.2;

Temp(1,:) = mean(RMSE_Capa,2)/0.0245;
Temp(2,:) = mean(RMSE_Life,2)/237.1346;
Temp(3,:) = mean(RMSE_ERate,2)/0.0238;
Temp(4,:) = mean(RMSE_CoChR,2)/0.0906;
Temp(5,:) = mean(RMSE_MipV,2)/0.1566;
Temp(6,:) = mean(RMSE_PlatfCapa,2)/0.0438;

MyTemp = mean(Temp,1);

% Plot the aggregated normalized RMSE across voltage setpoints
figure(1),hold on,plot(MyTemp,'-o'),box on,grid on