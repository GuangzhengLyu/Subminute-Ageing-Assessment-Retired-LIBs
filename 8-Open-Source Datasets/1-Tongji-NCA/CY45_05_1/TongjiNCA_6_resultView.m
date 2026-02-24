clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 1-Tongji-NCA
%%% This script: Load the Tongji-NCA single-cycle dataset and the saved PLSR
%%% test predictions (Y_Test), compute RMSE distributions over repeated runs,
%%% visualize ground truth vs predicted values for key indicators, and save
%%% pointwise prediction errors (ground truth minus prediction) to a MAT-file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load('OneCycle_TongjiNCA.mat')

%% Sample construction (ground-truth labels)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full available discharge-capacity trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (as stored in OneCycle)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Result summary: RMSE across repeated runs
% NOTE: The loop bounds and file name are kept exactly as in the original script
for i = 14:14
    % Load saved PLSR predictions (Y_Test has size: [CountRP, 6, Nsamples])
    load('./PLSR_Result_1_70_Y_Test_14_TongjiNCA.mat')
    for j = 1:100
        % RMSE for each indicator in run j
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

%% Visualization and pointwise error export (uses j from the loop above)
% NOTE: These lines intentionally use the final j value after the RMSE loop
% (kept exactly as in the original script)
figure(1),hold on,plot(Capa,squeeze(Y_Test(j,1,:)),'o'),ErCA_1 = Capa-squeeze(Y_Test(j,1,:));
figure(2),hold on,plot(Life,squeeze(Y_Test(j,2,:)),'s'),ErLI_1 = Life-squeeze(Y_Test(j,2,:));
figure(3),hold on,plot(ERate,squeeze(Y_Test(j,3,:)),'>'),ErER_1 = ERate-squeeze(Y_Test(j,3,:));
figure(4),hold on,plot(CoChRate,squeeze(Y_Test(j,4,:)),'d'),ErCR_1 = CoChRate-squeeze(Y_Test(j,4,:));
figure(5),hold on,plot(MindVolt,squeeze(Y_Test(j,5,:)),'^'),ErMV_1 = MindVolt-squeeze(Y_Test(j,5,:));
figure(6),hold on,plot(PlatfCapa,squeeze(Y_Test(j,6,:)),'>'),ErPC_1 = PlatfCapa-squeeze(Y_Test(j,6,:)); 

% Save pointwise errors for downstream statistics/plots
save Error_1.mat ErCA_1 ErLI_1 ErER_1 ErCR_1 ErMV_1 ErPC_1