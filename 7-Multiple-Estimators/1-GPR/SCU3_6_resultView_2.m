clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% This script: Load SCU3 Dataset #2, compute normalized ground-truth health
%%% indicators, then evaluate repeated GPR predictions saved per voltage
%%% setpoint. RMSE is summarized across repetitions and plotted versus the
%%% terminal-voltage grid. The mean RMSE curves are saved to MAT-file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load('../../OneCycle_2.mat')

%% Sample construction and health-indicator scaling
% Filter samples by the ending step flag and compute:
% (1) capacity-based SOH, (2) life (cycle count), and (3-6) expanded indicators.
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

        % Unified health-indicator scaling (consistent with training scripts)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% Result evaluation (RMSE across repetitions and setpoints)
% For each terminal-voltage setpoint index i (1..13), load repeated predictions
% Y_Test (100 repeats) and compute RMSE for each task and repeat.
for i = 1:13
    currentFile = sprintf('GPR_Result_2_60_Y_Test_%d.mat',i);
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

%% Plot RMSE versus terminal-voltage grid and save mean curves
% Terminal-voltage grid (13 setpoints, 3.0 V to 4.2 V in steps of 0.1 V)
Uter = 3.0:0.1:4.2;

% Task 1: capacity-based SOH
Temp = RMSE_Capa;
[1,mean(Temp(13,:)),std(Temp(13,:))]
figure(1),hold on,plot(Uter,mean(Temp,2),'-o')
M_RMSE(1,:) = mean(Temp,2);

% Task 2: life (cycle count)
Temp = RMSE_Life;
[2,mean(Temp(13,:)),std(Temp(13,:))]
figure(2),hold on,plot(Uter,mean(Temp,2),'-s')
M_RMSE(2,:) = mean(Temp,2);

% Task 3: energy efficiency
Temp = RMSE_ERate;
[3,mean(Temp(13,:)),std(Temp(13,:))]
figure(3),hold on,plot(Uter,mean(Temp,2),'-d')
M_RMSE(3,:) = mean(Temp,2);

% Task 4: constant-current charge rate
Temp = RMSE_CoChR;
[4,mean(Temp(13,:)),std(Temp(13,:))]
figure(4),hold on,plot(Uter,mean(Temp,2),'->')
M_RMSE(4,:) = mean(Temp,2);

% Task 5: mid-point voltage
Temp = RMSE_MipV;
[5,mean(Temp(13,:)),std(Temp(13,:))]
figure(5),hold on,plot(Uter,mean(Temp,2),'-^')
M_RMSE(5,:) = mean(Temp,2);

% Task 6: platform discharge capacity
Temp = RMSE_PlatfCapa;
[6,mean(Temp(13,:)),std(Temp(13,:))]
figure(6),hold on,plot(Uter,mean(Temp,2),'-<')
M_RMSE(6,:) = mean(Temp,2);

% Save mean RMSE curves (6 tasks × 13 setpoints)
save M_RMSE_GPR_2.mat M_RMSE