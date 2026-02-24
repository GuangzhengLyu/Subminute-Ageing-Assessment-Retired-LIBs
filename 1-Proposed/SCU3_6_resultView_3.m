clear
clc
% close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Load SCU3 Dataset #3, build normalized ground-truth outputs,
%%% evaluate saved PLSR predictions across voltage setpoints and repetitions,
%%% compute RMSE for each target, and visualize prediction performance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../OneCycle_3.mat')

%% Sample construction
% Filter samples by the ending step flag and extract normalized targets:
% capacity, life, and expanded health indicators.
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

        % Normalized capacity (relative scaling)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (normalized with fixed references)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result evaluation
% Load prediction results for each setpoint index and compute RMSE over runs
% (this script evaluates j = 1:2 as in the original code).
for i = 1:13
    currentFile = sprintf('./Result/PLSR_Result_3_50_Y_Test_%d.mat',i);
    load(currentFile)

    for j = 1:2
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Example scatter plots and error vectors (last loaded Y_Test and the current j value)
% These plots compare ground truth versus predicted values for each target.
figure(7),hold on,plot(Capa,squeeze(Y_Test(j,1,:)),'o'),ErCA_3 = Capa-squeeze(Y_Test(j,1,:));
figure(8),hold on,plot(Life,squeeze(Y_Test(j,2,:)),'s'),ErLI_3 = Life-squeeze(Y_Test(j,2,:));
figure(9),hold on,plot(ERate,squeeze(Y_Test(j,3,:)),'>'),ErER_3 = ERate-squeeze(Y_Test(j,3,:));
figure(10),hold on,plot(CoChRate,squeeze(Y_Test(j,4,:)),'d'),ErCR_3 = CoChRate-squeeze(Y_Test(j,4,:));
figure(11),hold on,plot(MindVolt,squeeze(Y_Test(j,5,:)),'^'),ErMV_3 = MindVolt-squeeze(Y_Test(j,5,:));
figure(12),hold on,plot(PlatfCapa,squeeze(Y_Test(j,6,:)),'>'),ErPC_3 = PlatfCapa-squeeze(Y_Test(j,6,:)); 

% save Error_3.mat ErCA_3 ErLI_3 ErER_3 ErCR_3 ErMV_3 ErPC_3

%% RMSE trend versus terminal voltage
% Uter corresponds to the 13 terminal-voltage setpoints used in feature extraction.
Uter = 3.0:0.1:4.2;

Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,plot(Uter,mean(Temp,2),'o-'),axis([2.9,4.3,0,0.05])

Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,plot(Uter,mean(Temp,2),'s-'),axis([2.9,4.3,0,240])

Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,plot(Uter,mean(Temp,2),'d-'),axis([2.9,4.3,0,0.025])

Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,plot(Uter,mean(Temp,2),'>-'),axis([2.9,4.3,0,0.1])

Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,plot(Uter,mean(Temp,2),'^-'),axis([2.9,4.3,0,0.16])

Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,plot(Uter,mean(Temp,2),'<-'),axis([2.9,4.3,0.01,0.11])