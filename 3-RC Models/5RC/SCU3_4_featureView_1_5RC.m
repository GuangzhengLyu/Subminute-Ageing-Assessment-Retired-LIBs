clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Relaxation-Voltage Feature Extraction
%%% This script: Extract 5RC features (Uoc, R0, R1–R5, C1–C5) from SCU3 Dataset #1
%%% - Fit a 5-term exponential relaxation model at 13 voltage setpoints (3.0–4.2 V)
%%% - Use dataset-specific index sets to select the initial bias for Uoc (Vset - Bias)
%%% - Reorder fitted exponential terms to enforce amplitude ranking consistency
%%% - Save/plot extracted ECM-like parameters versus original capacity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
load("../../OneCycle_1.mat")

% Index sets used to select Bias in a nested manner (coarse-to-fine screening)
Set_Not05 = [1,2,3,5,6,10,12,20,21,23,26,27,28,29,30,32,35,37,40,41,46,47,48,49,51,52,57,58,61,64,67,70,72,74,75,77,78,82,84,85,87,92,94,95,98,100];
Set_Not03 = [6,10,12,20,23,27,28,29,30,35,41,46,47,57,61,70,74,75,77,78,85,87,95,98];
Set_Not08 = [10,12,27,28,29,35,41,46,57,61,70,74,75,77,85];
Set_Not06 = [10,12,28,35,41,46,57,61,70,74,77];
Set_Not04 = [10,12,28,35,41,46,57,61,70,74,77];
Set_Not07 = [10,28,35,41,46,57,61,70,74,77];
Set_Not09 = [28,35,41,46,57,61,74,77];
Set_Not02 = [28,35,41,46,57,61,74,77];
Set_Not10 = [28,41,46,57,61,74,77];
Set_Not01 = [28,41,57,61,74,77];

tic
CountData = 0;
CountFit = 0;

for IndexData = 1:length(OneCycle)
    % figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % RUL proxy: first cycle index where discharge capacity drops below 2.5 Ah
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity (Ah)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation-voltage feature extraction (13 setpoints)
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
        R5(CountData,1:13) = 0;
        C5(CountData,1:13) = 0;

        % Skip samples in the exclusion set
        if ~ismember(IndexData,Set_Not01)
            CountFit = CountFit+1;
            CountRX = 0;
            MRMSE = 0;
            MyVoltege = 3.0;

            for Vset = MyVoltege:0.1:4.2
                CountRX = CountRX+1;
                LengthDrop = 10;   % discard early points after step switch
                LengthCops = 0;

                % Locate relaxation segment by step encoding
                IndexRX = find(OneCycle(IndexData).Steps == single((Vset-2.9)/0.1*2));
                Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop:end));
                PointRX(CountRX,1) = IndexRX(1);
                PointRX(CountRX,2) = IndexRX(end);

                %% Model fitting for relaxation voltage
                MyData = Vrlx{CountData,1};
                if MyData(1)-Vset(1) < 0
                    EndValueRX(CountData,CountRX) = MyData(end);

                    TimeInP = LengthCops+1:LengthCops+length(MyData);

                    % Select Bias for initial Uoc guess (Vset - Bias), using nested index sets
                    if ismember(IndexData, Set_Not05) && ~ismember(IndexData,Set_Not03)
                        Bias = 0.3;
                    elseif ismember(IndexData, Set_Not03) && ~ismember(IndexData,Set_Not08)
                        Bias = 0.8;
                    elseif ismember(IndexData, Set_Not08) && ~ismember(IndexData,Set_Not06)
                        Bias = 0.6;
                    elseif ismember(IndexData, Set_Not06) && ~ismember(IndexData,Set_Not04)
                        Bias = 0.4;
                    elseif ismember(IndexData, Set_Not04) && ~ismember(IndexData,Set_Not07)
                        Bias = 0.7;
                    elseif ismember(IndexData, Set_Not07) && ~ismember(IndexData,Set_Not09)
                        Bias = 0.9;
                    elseif ismember(IndexData, Set_Not09) && ~ismember(IndexData,Set_Not02)
                        Bias = 0.2;
                    elseif ismember(IndexData, Set_Not02) && ~ismember(IndexData,Set_Not10)
                        Bias = 1;
                    elseif ismember(IndexData, Set_Not10)
                        Bias = 0.1;
                    else
                        Bias = 0.5;
                    end

                    % 5RC relaxation model: sum of 5 exponentials + steady-state Uoc
                    beta0=[0.1 10 0.1 10 0.1 10 0.1 10 0.1 10 Vset-Bias];

                    mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                                        beta(3)*exp(-x./beta(4))+...
                                        beta(5)*exp(-x./beta(6))+...
                                        beta(7)*exp(-x./beta(8))+...
                                        beta(9)*exp(-x./beta(10))+...
                                        beta(11);

                    [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
                    [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);

                    % Running mean RMSE across the 13 setpoints for this sample
                    RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
                    MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
                    MMRMSE_Fit(CountFit) = MRMSE;

                    % Enforce consistent ordering of exponential amplitudes (descending)
                    for i = 1:10
                        if Beta(CountData,7) < Beta(CountData,9)
                            Temp79 = Beta(CountData,7);
                            Temp810 = Beta(CountData,8);
                            Beta(CountData,7) = Beta(CountData,9);
                            Beta(CountData,8) = Beta(CountData,10);
                            Beta(CountData,9) = Temp79;
                            Beta(CountData,10) = Temp810;
                        end
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

                    %% Feature construction (convert fitted voltage terms to ECM-like parameters)
                    Uoc(CountData,CountRX) = Beta(CountData,11);

                    % Current used for mapping voltage terms to resistances (A)
                    % Note: denominator 1.75 is treated as a fixed current scale in this script.
                    R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3)-Beta(CountData,5)-Beta(CountData,7)-Beta(CountData,9)-Beta(CountData,11))/1.75;

                    R1(CountData,CountRX)  = abs(Beta(CountData,1)/1.75);
                    C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/10/R1(CountData,CountRX));

                    R2(CountData,CountRX)  = abs(Beta(CountData,3)/1.75);
                    C2(CountData,CountRX)  = abs((Beta(CountData,4)+LengthDrop)/10/R2(CountData,CountRX));

                    R3(CountData,CountRX)  = abs(Beta(CountData,5)/1.75);
                    C3(CountData,CountRX)  = abs((Beta(CountData,6)+LengthDrop)/10/R3(CountData,CountRX));

                    R4(CountData,CountRX)  = abs(Beta(CountData,7)/1.75);
                    C4(CountData,CountRX)  = abs((Beta(CountData,8)+LengthDrop)/10/R4(CountData,CountRX));

                    R5(CountData,CountRX)  = abs(Beta(CountData,9)/1.75);
                    C5(CountData,CountRX)  = abs((Beta(CountData,10)+LengthDrop)/10/R5(CountData,CountRX));
                end
            end
        end
    end
end
toc

mean(MMRMSE_Fit)

% figure(1),hold on,plot(Capa,EndValueRX(:,13),'o')

%% Plot extracted features versus original capacity for each setpoint (13 -> 1)
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
    figure(11),clf,hold on,plot(Capa,R5(:,indexRX),'*')
    figure(12),clf,hold on,plot(Capa,C5(:,indexRX),'+')
end