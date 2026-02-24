clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment
%%% File:    M_RMSE_SVR_1.m
%%% Purpose: Post-process SVR results (SCU3 Dataset #1) by computing RMSE across
%%%          6 ageing-assessment tasks over 13 voltage setpoints and 100 repeats.
%%% Outputs: M_RMSE (6 x 13) saved to "M_RMSE_SVR_1.mat", where each row is the
%%%          mean RMSE vs. setpoint for one task.
%%% Notes:   This script expects "SVR_Result_1_70_Y_Test_%d.mat" files (d=1..13)
%%%          in the current working directory. Each file should contain Y_Test
%%%          with size (100 x 6 x Nsamples).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load SCU3 Dataset #1
load('../../OneCycle_1.mat')

%% Build ground-truth targets (6 tasks)
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Life (RUL proxy): first cycle index where discharge capacity drops below 2.5 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Capacity-based SOH (normalized)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;

        % Expanded performance indicators (normalized)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% RMSE evaluation across setpoints and repeats
for i = 1:13
    currentFile = sprintf('SVR_Result_1_70_Y_Test_%d.mat',i);
    load(currentFile)

    % RMSE per repeat (j=1..100) for each task at setpoint i
    for j = 1:100
        RMSE_Capa(i,j)      = sqrt(mean((Capa      - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j)      = sqrt(mean((Life      - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j)     = sqrt(mean((ERate     - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j)     = sqrt(mean((CoChRate  - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j)      = sqrt(mean((MindVolt  - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Aggregate and visualize mean RMSE vs. voltage setpoint
Uter = 3.0:0.1:4.2;  % 13 setpoints

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

%% Save summary RMSE matrix
save M_RMSE_SVR_1.mat M_RMSE