clear
clc
% close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123 (MITA123_3)
%%% This script: Load ground-truth health indicators, read saved PLSR
%%% predictions, compute RMSE across repeated runs, and visualize
%%% prediction-vs-ground-truth scatter with error vectors.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load('OneCycle_MITA123_3.mat')

%% Sample construction (ground-truth targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life: total number of available discharge-capacity samples (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Result summary
% Load saved PLSR predictions and compute RMSE for each repeated run
for i = 14:14
    load('./PLSR_Result_1_80_Y_Test_14_MITA123.mat')
    for j = 1:100
        RMSE_Capa(i,j) = sqrt(mean((Capa - squeeze(Y_Test(j,1,:))).^2));
        RMSE_Life(i,j) = sqrt(mean((Life - squeeze(Y_Test(j,2,:))).^2));
        RMSE_ERate(i,j) = sqrt(mean((ERate - squeeze(Y_Test(j,3,:))).^2));
        RMSE_CoChR(i,j) = sqrt(mean((CoChRate - squeeze(Y_Test(j,4,:))).^2));
        RMSE_MipV(i,j) = sqrt(mean((MindVolt - squeeze(Y_Test(j,5,:))).^2));
        RMSE_PlatfCapa(i,j) = sqrt(mean((PlatfCapa - squeeze(Y_Test(j,6,:))).^2));
    end
end

% Visualization: scatter of ground truth vs prediction (uses the last j)
% Note: Variables Er* store per-sample prediction error vectors.
figure(1),hold on,plot(Capa,squeeze(Y_Test(j,1,:)),'o'),ErCA_1 = Capa-squeeze(Y_Test(j,1,:));
figure(2),hold on,plot(Life,squeeze(Y_Test(j,2,:)),'s'),ErLI_1 = Life-squeeze(Y_Test(j,2,:));
figure(3),hold on,plot(ERate,squeeze(Y_Test(j,3,:)),'>'),ErER_1 = ERate-squeeze(Y_Test(j,3,:));
figure(4),hold on,plot(CoChRate,squeeze(Y_Test(j,4,:)),'d'),ErCR_1 = CoChRate-squeeze(Y_Test(j,4,:));
figure(5),hold on,plot(MindVolt,squeeze(Y_Test(j,5,:)),'^'),ErMV_1 = MindVolt-squeeze(Y_Test(j,5,:));
figure(6),hold on,plot(PlatfCapa,squeeze(Y_Test(j,6,:)),'>'),ErPC_1 = PlatfCapa-squeeze(Y_Test(j,6,:));