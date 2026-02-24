clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Fit multi-exponential relaxation-voltage model at multiple
%%% terminal-voltage setpoints (3.0–4.2 V) to extract RC parameters
%%% (Uoc, R0, R1, C1, R2, C2, R3, C3) from SCU3 Dataset #3.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
load("../../OneCycle_3.mat")

tic
CountData = 0;
CountFit = 0;
for IndexData = 1:length(OneCycle)
    % figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation-voltage initialization
        % Pre-allocate per-sample RC feature arrays (13 setpoints)
        Uoc(CountData,1:13) = 0;
        R0(CountData,1:13) = 0;
        R1(CountData,1:13) = 0;
        C1(CountData,1:13) = 0;
        R2(CountData,1:13) = 0;
        C2(CountData,1:13) = 0;
        R3(CountData,1:13) = 0;
        C3(CountData,1:13) = 0;
        
        % Exclude selected outlier samples from fitting
        if ~ismember(IndexData,[34])
            CountFit = CountFit+1;
            CountRX = 0;
            MRMSE = 0;
            MyVoltege = 3.0;
            for Vset = MyVoltege:0.1:4.2
                CountRX = CountRX+1;
                LengthDrop = 10;
                LengthCops = 0;
                
                % Locate relaxation segment indices at the current voltage setpoint
                IndexRX = find(OneCycle(IndexData).Steps == single((Vset-2.9)/0.1*2));
                Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop:end));
                PointRX(CountRX,1) = IndexRX(1);
                PointRX(CountRX,2) = IndexRX(end);
    
                %% Health feature construction (nonlinear fitting on relaxation trace)
                MyData = Vrlx{CountData,1};
                if MyData(1)-Vset(1) < 0
                    EndValueRX(CountData,CountRX) = MyData(end);

                    % Time index for fitting (relative sample index in the relaxation window)
                    TimeInP = LengthCops+1:LengthCops+length(MyData);

                    % Sample-specific bias for initial parameter guess (empirical stabilization)
                    if ismember(IndexData, [7,13,19,28,32,35,40,41,63,64,69,70,78,80,82,86,89,94,95,96,...
                                            110,111,112,117,120,123,124,127,130,134,158,163,167,176,184,191,195,196,...
                                            209,212,216,217,219,221,222,224,236,240,242,246,247,253,258,259,272,275,278,282,298,300,...
                                            302,303,309,314,317,320,325,329,341,342,347,348,351,356,369,370,372,375,376,385,386,387,...
                                            402,406,407,412,413,414,416,423,424,430,431])
                        Bias = 0.3;
                    elseif ismember(IndexData, [12,46,52,57,58,59,65,77,114,129,137,174,180,182,199,...
                                                206,208,211,214,218,231,235,237,252,254,256,268,270,273,281,306,310,312,321,322,323,327,328,358,365,382,395])
                        Bias = 0.8;
                    elseif ismember(IndexData, [22,84,93,103,177,263,289,311,362,371,374,379,408,418])
                        Bias = 0.6;
                    elseif ismember(IndexData, [53,76,164,248,274])
                        Bias = 0.4;
                    elseif ismember(IndexData, [161,239,280,350])
                        Bias = 0.7;
                    elseif ismember(IndexData, [])
                        Bias = 0.2;
                    elseif ismember(IndexData, [156])
                        Bias = 0.9;
                    elseif ismember(IndexData, [165])
                        Bias = 0.1;
                    elseif ismember(IndexData, [])
                        Bias = 0;
                    elseif ismember(IndexData, [324])
                        Bias = 1;
                    else
                        Bias = 0.5;
                    end
                    beta0=[0.1 10 0.1 10 0.1 10 Vset-Bias];

                    % Three-exponential relaxation model with steady-state offset beta(7)
                    mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                                        beta(3)*exp(-x./beta(4))+...
                                        beta(5)*exp(-x./beta(6))+...
                                        beta(7);

                    % Nonlinear regression and prediction confidence interval (used for RMSE)
                    [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
                    [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);
                    RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
                    MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
                    MMRMSE_Fit(CountFit) = MRMSE;

                    % figure(7),hold on,box on
                    % plot([MyData-Vset],'linewidth', 2,'color',[0 CountRX/13 1-CountRX/26])
                    % plot([Y-Vset],'--','linewidth', 2,'color',[0 0 0])
                    % title('MRMSE=', MRMSE)

                    % Enforce consistent ordering of exponential amplitudes/time constants
                    if Beta(CountData,3) < Beta(CountData,5)
                        Temp35 = Beta(CountData,3);
                        Temp46 = Beta(CountData,4);
                        Beta(CountData,3) = Beta(CountData,5);
                        Beta(CountData,4) = Beta(CountData,6);
                        Beta(CountData,5) = Temp35;
                        Beta(CountData,6) = Temp46;
                    end
                    if Beta(CountData,1) < Beta(CountData,3)
                        Temp13 = Beta(CountData,1);
                        Temp24 = Beta(CountData,2);
                        Beta(CountData,1) = Beta(CountData,3);
                        Beta(CountData,2) = Beta(CountData,4);
                        Beta(CountData,3) = Temp13;
                        Beta(CountData,4) = Temp24;
                    end
                    if Beta(CountData,3) < Beta(CountData,5)
                        Temp35 = Beta(CountData,3);
                        Temp46 = Beta(CountData,4);
                        Beta(CountData,3) = Beta(CountData,5);
                        Beta(CountData,4) = Beta(CountData,6);
                        Beta(CountData,5) = Temp35;
                        Beta(CountData,6) = Temp46;
                    end
                    if Beta(CountData,1) < Beta(CountData,3)
                        Temp13 = Beta(CountData,1);
                        Temp24 = Beta(CountData,2);
                        Beta(CountData,1) = Beta(CountData,3);
                        Beta(CountData,2) = Beta(CountData,4);
                        Beta(CountData,3) = Temp13;
                        Beta(CountData,4) = Temp24;
                    end

                    %% RC feature extraction
                    % Convert fitted parameters beta to RC elements at the current setpoint
                    Uoc(CountData,CountRX) = Beta(CountData,7);
                    R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3)-Beta(CountData,5)-Beta(CountData,7))/1.75;
                    R1(CountData,CountRX)  = abs(Beta(CountData,1)/1.75);
                    C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/10/R1(CountData,CountRX));
                    R2(CountData,CountRX)  = abs(Beta(CountData,3)/1.75);
                    C2(CountData,CountRX)  = abs((Beta(CountData,4)+LengthDrop)/10/R2(CountData,CountRX));
                    R3(CountData,CountRX)  = abs(Beta(CountData,5)/1.75);
                    C3(CountData,CountRX)  = abs((Beta(CountData,6)+LengthDrop)/10/R3(CountData,CountRX));
                end
            end
        end
    end
end
toc

% Average fitting error across fitted samples (MRMSE over setpoints)
mean(MMRMSE_Fit)

% figure(1),hold on,plot(Capa,EndValueRX(:,1),'o')

%% Quick inspection plots (RC parameters vs capacity, per setpoint index)
for i = 13:-1:1
    indexRX = i;

    figure(1),clf,hold on,plot(flip(Capa),Uoc(:,indexRX),'o')
    figure(2),clf,hold on,plot(flip(Capa),R0(:,indexRX),'d')
    figure(3),clf,hold on,plot(flip(Capa),R1(:,indexRX),'<')
    figure(4),clf,hold on,plot(flip(Capa),C1(:,indexRX),'>')
    figure(5),clf,hold on,plot(flip(Capa),R2(:,indexRX),'*')
    figure(6),clf,hold on,plot(flip(Capa),C2(:,indexRX),'+')
    figure(7),clf,hold on,plot(flip(Capa),R3(:,indexRX),'*')
    figure(8),clf,hold on,plot(flip(Capa),C3(:,indexRX),'+')
end