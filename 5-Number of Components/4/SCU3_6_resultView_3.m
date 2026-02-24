clear
clc
% close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: PLSR Latent Component Sensitivity Study
%%% This script: Load SCU3 Dataset #3, assemble normalized health indicators,
%%% then read saved PLSR4 prediction results across voltage setpoints to
%%% compute RMSE statistics and visualize prediction vs. ground truth trends.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../../OneCycle_3.mat')

%% Sample construction and indicator normalization
% Filter samples by the ending step flag and extract life, capacity, and
% expanded health indicators (already scaled into comparable forms here).
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
        
        % Capacity-based SOH proxy (normalized by nominal capacity)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;
        
        % Expanded health indicators (normalized by fixed reference bounds)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result aggregation (RMSE over repetitions and voltage setpoints)
% For each voltage setpoint index i (1..13), load the saved predictions
% (Y_Test: repetitions x outputs x samples), then compute RMSE per repetition.
for i = 1:13
    currentFile = sprintf('./PLSR4_Result_3_50_Y_Test_%d.mat',i);
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

%% Example scatter plots and residual vectors
% Note: j is taken from the last loop value (j = 100) as in the original code.
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
   
%% Voltage grid and aggregated normalized RMSE summary
% Uter defines the voltage axis (not explicitly used in plotting below).
Uter = 3.0:0.1:4.2;

% Normalize mean RMSE curves by fixed scaling constants (per-indicator ranges)
Temp(1,:) = mean(RMSE_Capa,2)/0.0246;
Temp(2,:) = mean(RMSE_Life,2)/237.7643;
Temp(3,:) = mean(RMSE_ERate,2)/0.0240;
Temp(4,:) = mean(RMSE_CoChR,2)/0.0912;
Temp(5,:) = mean(RMSE_MipV,2)/0.1581;
Temp(6,:) = mean(RMSE_PlatfCapa,2)/0.0439;

% Average across all indicators to obtain a single sensitivity curve
MyTemp = mean(Temp,1);

% Plot aggregated sensitivity metric across setpoints
figure(1),hold on,plot(MyTemp,'-o'),box on,grid on