clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: PLSR Latent Component Sensitivity Study
%%% This script: Load SCU3 Dataset #1 single-cycle records, reconstruct the
%%% ground-truth targets (capacity, life, and expanded health indicators),
%%% then evaluate PLSR3 prediction results saved per voltage setpoint.
%%% The script computes RMSE across 100 repeats for each setpoint, forms a
%%% normalized multi-metric score, and plots the sensitivity trend versus
%%% setpoint index.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SCU3 Dataset #1
% Load structured single-cycle dataset
load('../../OneCycle_1.mat')

%% Reconstruct ground-truth targets (health indicators)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH proxy (scaled by nominal capacity)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (scaled to the unified [0,1]-type form used in this study)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Results aggregation (PLSR3)
% For each setpoint index i = 1..13:
%   - load Y_Test saved by the PLSR3 script
%   - compute RMSE for each output dimension across 100 repeats
for i = 1:13
    currentFile = sprintf('./PLSR3_Result_1_70_Y_Test_%d.mat',i);
    load(currentFile)

    for j = 1:100
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Example scatter plots and per-output errors
% Note: j remains from the loop above; it will be 100 at this point.
% These plots use the last repetition's predictions from the last loaded Y_Test.
figure(7), hold on, plot(Capa,      squeeze(Y_Test(j,1,:)),'o')
ErCA_1 = Capa      - squeeze(Y_Test(j,1,:));

figure(8), hold on, plot(Life,      squeeze(Y_Test(j,2,:)),'s')
ErLI_1 = Life      - squeeze(Y_Test(j,2,:));

figure(9), hold on, plot(ERate,     squeeze(Y_Test(j,3,:)),'>')
ErER_1 = ERate     - squeeze(Y_Test(j,3,:));

figure(10), hold on, plot(CoChRate, squeeze(Y_Test(j,4,:)),'d')
ErCR_1 = CoChRate  - squeeze(Y_Test(j,4,:));

figure(11), hold on, plot(MindVolt, squeeze(Y_Test(j,5,:)),'^')
ErMV_1 = MindVolt  - squeeze(Y_Test(j,5,:));

figure(12), hold on, plot(PlatfCapa, squeeze(Y_Test(j,6,:)),'>')
ErPC_1 = PlatfCapa - squeeze(Y_Test(j,6,:));

%% Normalized multi-metric score
% Convert each RMSE curve into a normalized scale using pre-defined factors,
% then average across the six outputs to obtain a single sensitivity score.
Uter = 3.0:0.1:4.2; %#ok<NASGU>

Temp(1,:) = mean(RMSE_Capa,2)/0.0450;
Temp(2,:) = mean(RMSE_Life,2)/67.7949;
Temp(3,:) = mean(RMSE_ERate,2)/0.0180;
Temp(4,:) = mean(RMSE_CoChR,2)/0.0293;
Temp(5,:) = mean(RMSE_MipV,2)/0.0430;
Temp(6,:) = mean(RMSE_PlatfCapa,2)/0.1089;

MyTemp = mean(Temp,1);

%% Plot sensitivity trend across setpoint indices
figure(1), hold on, plot(MyTemp,'-o'), box on, grid on