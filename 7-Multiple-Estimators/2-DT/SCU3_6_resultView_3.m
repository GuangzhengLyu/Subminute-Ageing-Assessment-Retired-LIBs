clear
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Aggregate RMSE statistics for decision-tree (DT) predictions
%%% on SCU3 Dataset #3 across multiple relaxation-voltage setpoints, then
%%% plot mean RMSE versus terminal voltage and save the mean RMSE curves.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load('../../OneCycle_3.mat')

%% Ground-truth label construction (normalized indicator space)
% Rebuild life/capacity and expanded health indicators using the same
% definitions and scaling as in the corresponding DT training script.
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        % Life definition: first cycle where discharge capacity drops below 1.75 Ah
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 1.75)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 1.75));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        % Capacity-based SOH proxy (scaled by nominal capacity)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded health indicators (scaled to comparable ranges)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% RMSE aggregation across setpoints and repetitions
% Each result file stores Y_Test with dimensions:
% (repetition index, output dimension, sample index).
for i = 1:13
    currentFile = sprintf('DT_Result_3_50_Y_Test_%d.mat',i);
    load(currentFile)
    for j = 1:100
        % RMSE is computed in the (scaled) label space reconstructed above
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Plotting: mean RMSE vs relaxation terminal voltage
% Voltage setpoints are assumed to span 3.0 to 4.2 V with 0.1 V step (13 points).
Uter = 3.0:0.1:4.2;

Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,plot(Uter,mean(Temp,2),'-o')
M_RMSE(1,:) = mean(Temp,2);

Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,plot(Uter,mean(Temp,2),'-s')
M_RMSE(2,:) = mean(Temp,2);

Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,plot(Uter,mean(Temp,2),'-d')
M_RMSE(3,:) = mean(Temp,2);

Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,plot(Uter,mean(Temp,2),'->')
M_RMSE(4,:) = mean(Temp,2);

Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,plot(Uter,mean(Temp,2),'-^')
M_RMSE(5,:) = mean(Temp,2);

Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,plot(Uter,mean(Temp,2),'-<')
M_RMSE(6,:) = mean(Temp,2);

%% Save aggregated mean RMSE curves
% M_RMSE rows correspond to output dimensions 1..6, columns correspond to setpoints.
save M_RMSE_DT_3.mat M_RMSE