clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Relaxation-Voltage Feature Extraction
%%% This script: Extract 5RC features (Uoc, R0, R1–R5, C1–C5) from SCU3 Dataset #3
%%% - Fit a 5-term exponential relaxation model at 13 voltage setpoints (3.0–4.2 V)
%%% - Use nested index sets to choose Bias for the initial Uoc guess (Vset - Bias)
%%% - Reorder fitted exponential terms to enforce consistent amplitude ranking
%%% - Skip a predefined set of problematic samples (Set_Not01)
%%% - Plot extracted parameters versus original capacity for each setpoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
load("../../OneCycle_3.mat")

% Index sets for Bias selection (nested, coarse-to-fine)
Set_Not05 = [1,3,4,6,9,10,11,12,14,15,16,17,18,20,21,23,24,25,26,29,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,51,52,53,55,58,60,61,62,63,64,65,68,69,70,71,72,73,74,75,76,77,78,81,82,83,84,85,87,88,91,92,93,94,96,101,102,103,104,105,109,111,112,113,114,116,117,118,119,121,122,124,126,127,128,129,130,131,132,133,135,136,141,142,147,149,151,152,153,155,156,157,159,164,165,170,172,173,178,179,180,181,182,185,186,187,191,192,193,194,196,197,198,199,202,203,204,209,211,212,213,215,216,217,218,221,222,223,227,229,231,232,234,236,239,240,241,242,243,244,246,247,249,251,252,253,254,255,256,258,259,260,261,262,263,264,265,266,267,268,269,271,272,275,277,278,279,282,285,287,290,291,292,293,298,304,305,306,307,308,309,311,312,313,315,317,318,319,320,321,322,323,325,326,328,330,331,337,338,340,343,344,348,349,350,351,355,356,357,358,359,362,363,364,367,368,370,372,373,374,377,378,379,382,385,386,387,391,392,393,394,395,396,398,402,403,404,405,406,407,408,409,410,411,412,413,414,415,417,422,424,427,428,429,430,431,433];
Set_Not03 = [4,6,9,10,11,14,15,16,17,18,20,21,23,25,32,34,35,36,38,40,41,42,48,49,51,52,53,55,58,60,61,62,64,65,70,71,73,75,77,78,82,83,84,85,87,91,93,94,96,101,103,104,105,109,111,112,113,114,116,117,118,122,124,126,127,128,129,130,131,133,135,136,147,151,155,159,164,165,170,178,179,180,181,182,185,186,191,192,193,194,196,198,199,203,204,212,213,215,216,217,218,222,229,232,234,236,241,243,244,247,249,251,253,255,258,259,261,262,265,267,268,269,271,275,277,278,279,282,285,287,291,292,293,304,305,306,308,311,313,315,317,318,319,320,323,326,328,330,348,349,355,356,358,363,367,368,370,372,373,377,378,382,385,386,391,392,393,394,395,396,398,402,403,404,405,406,407,409,412,415,417,424,428,429,430,431,433];
Set_Not08 = [6,9,10,11,14,15,16,17,18,20,23,25,32,34,35,36,38,41,42,49,51,52,53,55,58,60,61,62,64,65,71,73,75,78,83,84,85,93,94,96,103,104,105,109,111,112,113,114,116,117,118,122,124,126,127,129,131,135,147,155,164,179,180,181,182,185,186,191,192,193,194,199,203,204,212,213,215,216,217,218,222,229,232,236,241,243,244,247,249,251,253,255,256,258,259,262,265,268,269,275,277,278,282,285,287,291,292,293,304,306,308,311,313,315,318,320,326,328,330,349,356,358,363,368,372,373,377,378,382,385,386,391,392,394,395,396,398,402,403,404,405,406,407,409,415,417,428,429,430,431];
Set_Not06 = [6,9,10,11,14,15,16,17,18,25,34,35,36,38,49,52,53,55,58,60,61,64,65,71,73,75,83,84,94,103,104,105,109,111,112,113,116,117,122,124,129,164,180,181,185,186,192,193,194,204,212,213,215,216,217,222,229,232,241,243,244,247,249,251,253,255,258,259,262,268,275,277,278,282,285,287,291,292,293,304,306,311,313,315,318,326,328,330,349,356,358,368,372,373,377,382,386,392,394,395,398,403,404,405,406,407,409,415,428,430];
Set_Not04 = [6,10,11,15,16,17,25,34,35,36,38,52,53,55,58,61,64,65,73,75,84,94,103,104,105,111,112,113,117,122,124,164,180,181,185,186,192,193,194,204,212,215,216,217,222,229,232,241,243,247,249,251,255,258,262,268,282,285,287,291,292,293,304,306,311,313,315,318,326,328,330,349,356,368,372,373,377,382,386,392,394,395,398,404,405,407,409,415,430];
Set_Not07 = [6,10,11,15,16,17,34,36,38,52,55,58,61,64,65,73,75,84,94,103,105,111,112,113,117,122,124,180,181,186,192,193,194,215,216,217,222,229,232,241,243,247,249,251,255,262,268,282,287,291,292,293,306,311,315,318,326,328,330,349,372,373,377,382,386,392,394,395,398,405,407,409,430];
Set_Not09 = [6,10,11,16,17,34,36,38,55,61,64,65,73,75,84,111,112,113,117,122,180,181,186,192,193,194,215,216,217,222,229,232,241,247,249,251,255,262,268,282,287,291,292,311,315,318,326,328,330,349,372,373,377,382,386,394,395,407,430];
Set_Not02 = [6,10,17,34,36,38,55,61,65,73,75,84,111,112,113,122,180,181,186,192,193,194,215,216,217,222,229,232,247,249,251,255,262,268,282,291,292,311,315,318,326,328,330,349,372,373,377,382,386,394,395,407];
Set_Not10 = [6,10,17,34,36,38,55,61,65,75,84,111,112,113,122,180,181,186,193,194,215,216,217,222,229,232,247,249,251,255,268,282,291,292,311,315,318,326,328,330,349,372,382,386,395,407];
Set_Not01 = [6,10,17,34,36,38,55,61,65,84,111,113,122,180,181,186,193,194,215,216,217,222,229,232,249,251,255,268,282,292,311,315,318,326,328,330,349,372,382,386];

tic
CountData = 0;
CountFit  = 0;

for IndexData = 1:length(OneCycle)
    % figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Original capacity (Ah)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation-voltage feature extraction (13 setpoints)
        Uoc(CountData,1:13) = 0;
        R0(CountData,1:13)  = 0;
        R1(CountData,1:13)  = 0;
        C1(CountData,1:13)  = 0;
        R2(CountData,1:13)  = 0;
        C2(CountData,1:13)  = 0;
        R3(CountData,1:13)  = 0;
        C3(CountData,1:13)  = 0;
        R4(CountData,1:13)  = 0;
        C4(CountData,1:13)  = 0;
        R5(CountData,1:13)  = 0;
        C5(CountData,1:13)  = 0;

        % Skip samples in the exclusion set
        if ~ismember(IndexData,Set_Not01)
            CountFit = CountFit+1;
            CountRX  = 0;
            MRMSE    = 0;
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