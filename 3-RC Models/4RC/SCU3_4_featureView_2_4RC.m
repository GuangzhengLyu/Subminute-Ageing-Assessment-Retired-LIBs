clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Extract 4RC relaxation features (Uoc, R0, R1, C1, R2, C2,
%%% R3, C3, R4, C4) from SCU3 Dataset #2 by fitting a 4-exponential decay
%%% model at multiple voltage setpoints (3.0–4.2 V).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load("../../OneCycle_2.mat")

tic
CountData = 0;
CountFit = 0;
for IndexData = 1:length(OneCycle)
    figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.1)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.1));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation voltage extraction (initialize per-setpoint feature arrays)
        Uoc(CountData,1:13) = 0;
        R0(CountData,1:13) = 0;
        R1(CountData,1:13) = 0;
        C1(CountData,1:13) = 0;
        R2(CountData,1:13) = 0;
        C2(CountData,1:13) = 0;
        R3(CountData,1:13) = 0;
        C3(CountData,1:13) = 0;
        R4(CountData,1:13) = 0;
        C4(CountData,1:13) = 0;
        
        % Optional exclusion list (kept empty here)
        if ~ismember(IndexData,[])
            CountFit = CountFit+1;
            CountRX = 0;
            MRMSE = 0;
            MyVoltege = 3.0;
            for Vset = MyVoltege:0.1:4.2
                CountRX = CountRX+1;
                LengthDrop = 10;
                LengthCops = 0;
                
                % Locate the relaxation segment index range for this voltage setpoint
                IndexRX = find(OneCycle(IndexData).Steps == single((Vset-2.9)/0.1*2));
                Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop:end));
                PointRX(CountRX,1) = IndexRX(1);
                PointRX(CountRX,2) = IndexRX(end);
                
                %% Health feature construction via nonlinear relaxation fitting
                MyData = Vrlx{CountData,1};
                if MyData(1)-Vset(1) < 0
                    EndValueRX(CountData,CountRX) = MyData(end);

                    % Time index for fitting (sample index domain)
                    TimeInP = LengthCops+1:LengthCops+length(MyData);

                    % Heuristic bias for initializing the open-circuit offset term
                    if ismember(IndexData, [6,11,15,17,22,40,41,43])
                        Bias = 0.3;
                    elseif ismember(IndexData, [1])
                        Bias = 0.8;
                    elseif ismember(IndexData, [4,29,32,35])
                        Bias = 0.6;
                    elseif ismember(IndexData, [])
                        Bias = 0.4;
                    elseif ismember(IndexData, [])
                        Bias = 0.7;
                    elseif ismember(IndexData, [])
                        Bias = 0.9;
                    elseif ismember(IndexData, [36])
                        Bias = 0.2;
                    else
                        Bias = 0.5;
                    end
                    beta0=[0.1 10 0.1 10 0.1 10 0.1 10 Vset-Bias];

                    % 4RC relaxation model: sum of four exponentials + offset
                    mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                                        beta(3)*exp(-x./beta(4))+...
                                        beta(5)*exp(-x./beta(6))+...
                                        beta(7)*exp(-x./beta(8))+...
                                        beta(9);

                    % Nonlinear least-squares fitting and prediction interval output
                    [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
                    [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);

                    % Fit error tracking (MRMSE averaged across setpoints)
                    RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
                    MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
                    MMRMSE_Fit(CountFit) = MRMSE;

                    % figure(7),hold on,box on
                    % plot([MyData-Vset],'linewidth', 2,'color',[0 CountRX/13 1-CountRX/26])
                    % plot([Y-Vset],'--','linewidth', 2,'color',[0 0 0])
                    % title('MRMSE=', MRMSE)
                    
                    % Enforce a consistent ordering of exponential amplitudes/time constants
                    for i = 1:10
                        if Beta(CountData,5) < Beta(CountData,7)
                            Temp57 = Beta(CountData,5);
                            Temp68 = Beta(CountData,6);
                            Beta(CountData,5) = Beta(CountData,7);
                            Beta(CountData,6) = Beta(CountData,8);
                            Beta(CountData,7) = Temp57;
                            Beta(CountData,8) = Temp68;
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
                    end

                    %% Feature construction (map fitted parameters to RC elements)
                    Uoc(CountData,CountRX) = Beta(CountData,9);
                    R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3)-Beta(CountData,5)-Beta(CountData,7)-Beta(CountData,9))/1.75;
                    R1(CountData,CountRX)  = abs(Beta(CountData,1)/1.75);
                    C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/10/R1(CountData,CountRX));
                    R2(CountData,CountRX)  = abs(Beta(CountData,3)/1.75);
                    C2(CountData,CountRX)  = abs((Beta(CountData,4)+LengthDrop)/10/R2(CountData,CountRX));
                    R3(CountData,CountRX)  = abs(Beta(CountData,5)/1.75);
                    C3(CountData,CountRX)  = abs((Beta(CountData,6)+LengthDrop)/10/R3(CountData,CountRX));
                    R4(CountData,CountRX)  = abs(Beta(CountData,7)/1.75);
                    C4(CountData,CountRX)  = abs((Beta(CountData,8)+LengthDrop)/10/R4(CountData,CountRX));
                end
            end
        end
    end
end
toc

% Mean fitting error across fitted samples
mean(MMRMSE_Fit)

% figure(1),hold on,plot(Capa,EndValueRX(:,13),'o')

%% Quick visualization across setpoints (capacity vs extracted parameters)
for i = 13:-1:1
    indexRX = i;

    figure(1),clf,hold on,plot(Capa,Uoc(:,indexRX),'o')
    figure(2),clf,hold on,plot(Capa,R0(:,indexRX),'d')
    figure(3),clf,hold on,plot(Capa,R1(:,indexRX),'<')
    figure(4),clf,hold on,plot(Capa,C1(:,indexRX),'>')
    figure(5),clf,hold on,plot(Capa,R2(:,indexRX),'*')
    figure(6),clf,hold on,plot(Capa,C2(:,indexRX),'+')
    figure(7),clf,hold on,plot(Capa,R3(:,indexRX),'*')
    figure(8),clf,hold on,plot(Capa,C3(:,indexRX),'+')
    figure(9),clf,hold on,plot(Capa,R4(:,indexRX),'*')
    figure(10),clf,hold on,plot(Capa,C4(:,indexRX),'+')
end