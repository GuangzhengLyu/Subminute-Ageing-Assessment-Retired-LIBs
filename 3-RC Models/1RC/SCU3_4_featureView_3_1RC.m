clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: RC Model Order Study for Feature Extraction and Ageing Assessment
%%% This script: Extract 1-RC relaxation features (Uoc, R0, R1, C1) from
%%% SCU3 Dataset #3 by fitting a single-exponential relaxation model at
%%% multiple voltage setpoints (3.0–4.2 V), then visualize feature trends
%%% versus initial capacity.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #3
% Load structured single-cycle dataset
load("../../OneCycle_3.mat")

tic
CountData = 0;
CountFit = 0;
for IndexData = 1:length(OneCycle)
    % figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Store the initial discharge capacity proxy for downstream plotting
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation voltage extraction
        % Pre-allocate per-sample feature arrays for 13 voltage setpoints
        Uoc(CountData,1:13) = 0;
        R0(CountData,1:13) = 0;
        R1(CountData,1:13) = 0;
        C1(CountData,1:13) = 0;

        % Optional sample exclusion hook (kept as-is)
        if ~ismember(IndexData,[])
            CountFit = CountFit+1;
            CountRX = 0;
            MRMSE = 0;
            MyVoltege = 3.0;

            % Sweep voltage setpoints (3.0–4.2 V, step 0.1 V)
            for Vset = MyVoltege:0.1:4.2
                CountRX = CountRX+1;
                LengthDrop = 10;
                LengthCops = 0;

                % Locate the relaxation segment corresponding to the target step
                IndexRX = find(OneCycle(IndexData).Steps == single((Vset-2.9)/0.1*2));
                Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop:end));
                PointRX(CountRX,1) = IndexRX(1);
                PointRX(CountRX,2) = IndexRX(end);

                %% Health feature construction
                % Fit a single-exponential relaxation model:
                %   V(t) = a*exp(-t/tau) + b
                MyData = Vrlx{CountData,1};
                if MyData(1)-Vset(1) < 0
                    EndValueRX(CountData,CountRX) = MyData(end);

                    % Build time index for regression (sample index domain)
                    TimeInP = LengthCops+1:LengthCops+length(MyData);

                    % Initial bias heuristic for robust fitting (kept as-is)
                    if ismember(IndexData, [191,309,310,317,320,323,325,337,348])
                        Bias = 0.3;
                    elseif ismember(IndexData, [155,307,356])
                        Bias = 0.8;
                    elseif ismember(IndexData, [])
                        Bias = 0.6;
                    elseif ismember(IndexData, [])
                        Bias = 0.4;
                    elseif ismember(IndexData, [])
                        Bias = 0.7;
                    else
                        Bias = 0.5;
                    end
                    beta0=[0.1 10 Vset-Bias];

                    mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                                        beta(3);

                    % Nonlinear least-squares fitting and confidence intervals
                    [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
                    [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);

                    % RMSE accumulation across voltage setpoints
                    RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
                    MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
                    MMRMSE_Fit(CountFit) = MRMSE;

                    % figure(7),hold on,box on
                    % plot([MyData-Vset],'linewidth', 2,'color',[0 CountRX/13 1-CountRX/26])
                    % plot([Y-Vset],'--','linewidth', 2,'color',[0 0 0])
                    % title('MRMSE=', MRMSE)

                    %% Feature extraction (1-RC equivalent)
                    % Map fitted parameters to Uoc/R0/R1/C1 using the same
                    % conventions as the original implementation.
                    Uoc(CountData,CountRX) = Beta(CountData,3);
                    R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3))/1.75;
                    R1(CountData,CountRX)  = abs(Beta(CountData,1)/1.75);
                    C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/10/R1(CountData,CountRX));
                end
            end
        end
    end
end
toc

% Mean fitting error across fitted samples
mean(MMRMSE_Fit)

% figure(1),hold on,plot(Capa,EndValueRX(:,1),'o')

%% Visualization
% Plot extracted features versus initial capacity for each voltage setpoint
for i = 13:-1:1
    indexRX = i;

    figure(1),clf,hold on,plot(flip(Capa),Uoc(:,indexRX),'o')
    figure(2),clf,hold on,plot(flip(Capa),R0(:,indexRX),'d')
    figure(3),clf,hold on,plot(flip(Capa),R1(:,indexRX),'<')
    figure(4),clf,hold on,plot(flip(Capa),C1(:,indexRX),'>')
end