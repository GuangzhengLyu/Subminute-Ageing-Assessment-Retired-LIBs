clear
clc
close all

% SCU3数据集#2
load('../OneCycle_2.mat')

CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh./3.5;
        % 拓展健康指标
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2)/89;
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2)/83;
        MindVolt(CountData,1) = (OneCycle(IndexData).Cycle.MindVoltV(1)-2.65)/(3.47-2.65);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1)/1.3;
    end
end

%% 结果展示
for InDexLV = 1:6
    for i = 1:13
        currentFile = sprintf('./%d/PLSR%d_Result_2_60_Y_Test_%d.mat',InDexLV,InDexLV,i);
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
    Error(1,InDexLV) = mean(mean(RMSE_Capa));
    Error(2,InDexLV) = mean(mean(RMSE_Life));
    Error(3,InDexLV) = mean(mean(RMSE_ERate));
    Error(4,InDexLV) = mean(mean(RMSE_CoChR));
    Error(5,InDexLV) = mean(mean(RMSE_MipV));
    Error(6,InDexLV) = mean(mean(RMSE_PlatfCapa));
end
figure,plot(Error(1,:),'o-'),axis([0,7,0.00181,0.00203])
figure,plot(Error(2,:),'s-'),axis([0,7,149,159])
figure,plot(Error(3,:),'d-'),axis([0,7,0.0073,0.0082])
figure,plot(Error(4,:),'>-'),axis([0,7,0.0298,0.0345])
figure,plot(Error(5,:),'^-'),axis([0,7,0.0395,0.045])
figure,plot(Error(6,:),'<-'),axis([0,7,0.034,0.044])
