clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Extract relaxation-voltage segments from SCU3 Dataset #1,
%%% fit a two-exponential relaxation model at multiple terminal voltages,
%%% and construct equivalent-circuit features (Uoc, R0, R1, C1, R2, C2).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #1
% Load structured single-cycle dataset
load("../OneCycle_1.mat")

%% Sample construction
% Filter samples by the ending step flag and extract life, original capacity,
% and relaxation-based features through nonlinear model fitting.
tic
CountData = 0;
CountFit = 0;
for IndexData = 1:length(OneCycle)
    % figure(7),clf
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 2.5 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity for downstream correlation plots
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        %% Relaxation-voltage extraction (initialization)
        % Pre-allocate per-sample feature containers for 13 voltage setpoints
        Uoc(CountData,1:13) = 0;
        R0(CountData,1:13) = 0;
        R1(CountData,1:13) = 0;
        C1(CountData,1:13) = 0;
        R2(CountData,1:13) = 0;
        C2(CountData,1:13) = 0;

        % IndexData filtering hook (currently empty set => always true)
        if ~ismember(IndexData,[])
            CountFit = CountFit+1;
            CountRX = 0;
            MRMSE = 0;

            % Start voltage for relaxation feature extraction
            MyVoltege = 3.0;

            % Sweep relaxation segments over multiple terminal voltages
            for Vset = MyVoltege:0.1:4.2
                CountRX = CountRX+1;

                % Drop initial samples to remove transient artifacts
                LengthDrop = 10;
                LengthCops = 0;

                % Locate indices associated with the target step (mapped from Vset)
                IndexRX = find(OneCycle(IndexData).Steps == single((Vset-2.9)/0.1*2));

                % Extract relaxation voltage trace after dropping initial points
                Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop:end));
                PointRX(CountRX,1) = IndexRX(1);
                PointRX(CountRX,2) = IndexRX(end);

                %% Health-feature construction (two-exponential relaxation fit)
                MyData = Vrlx{CountData,1};

                % Proceed only when the relaxation trace starts below the setpoint
                if MyData(1)-Vset(1) < 0

                    % Store the final relaxation value (used later for inspection)
                    EndValueRX(CountData,CountRX) = MyData(end);

                    % Time index used as the regressor for nonlinear fitting
                    TimeInP = LengthCops+1:LengthCops+length(MyData);

                    % Manual bias selection for specific samples (data-dependent correction)
                    if ismember(IndexData, [10,12,29,41,42,51,54,56,69,88,95,97,104])
                        Bias = 0.3;
                    elseif ismember(IndexData, [8,13,39])
                        Bias = 0.8;
                    elseif ismember(IndexData, [40])
                        Bias = 0.6;
                    elseif ismember(IndexData, [])
                        Bias = 0.4;
                    elseif ismember(IndexData, [])
                        Bias = 0.7;
                    elseif ismember(IndexData, [99])
                        Bias = 0.9;
                    else
                        Bias = 0.5;
                    end

                    % Initial parameter guess:
                    % beta = [A1, tau1, A2, tau2, offset], offset initialized near Vset-Bias
                    beta0=[0.1 10 0.1 10 Vset-Bias];

                    % Two-exponential relaxation model with constant offset
                    mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                                        beta(3)*exp(-x./beta(4))+...
                                        beta(5);

                    % Nonlinear least-squares fit and prediction interval computation
                    [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
                    [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);

                    % RMSE-based goodness-of-fit tracking across voltage setpoints
                    RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
                    MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
                    MMRMSE_Fit(CountFit) = MRMSE;

                    % figure(7),hold on,box on
                    % plot([MyData-Vset],'linewidth', 2,'color',[0 CountRX/13 1-CountRX/26])
                    % plot([Y-Vset],'-','linewidth', 2,'color',[0 0 0])
                    % title('MRMSE=', MRMSE)

                    % Enforce ordering: ensure the first exponential has larger amplitude
                    if Beta(CountData,1) < Beta(CountData,3)
                        Temp13 = Beta(CountData,1);
                        Temp24 = Beta(CountData,2);
                        Beta(CountData,1) = Beta(CountData,3);
                        Beta(CountData,2) = Beta(CountData,4);
                        Beta(CountData,3) = Temp13;
                        Beta(CountData,4) = Temp24;
                    end

                    %% Feature construction (ECM-like parameters)
                    % Uoc: fitted offset term
                    Uoc(CountData,CountRX) = Beta(CountData,5);

                    % R0: instantaneous drop mapped by nominal current (1.75 A)
                    R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3)-Beta(CountData,5))/1.75;

                    % R1/C1 and R2/C2: amplitudes and time constants mapped to RC pairs
                    R1(CountData,CountRX)  = abs(Beta(CountData,1)/1.75);
                    C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/10/R1(CountData,CountRX));
                    R2(CountData,CountRX)  = abs(Beta(CountData,3)/1.75);
                    C2(CountData,CountRX)  = abs((Beta(CountData,4)+LengthDrop)/10/R2(CountData,CountRX));
                end
            end
        end
    end
end
toc

% Report mean multi-voltage fit error across processed samples
mean(MMRMSE_Fit)

% Quick inspection: capacity versus final relaxation value at the last setpoint
figure(1),hold on,plot(Capa,EndValueRX(:,13),'o')

% Visualization of extracted features at a specific voltage-setpoint index
for i = 13:-1:13
    indexRX = i;

    figure(1),clf,hold on,plot(Capa,Uoc(:,indexRX),'o')
    figure(2),clf,hold on,plot(Capa,R0(:,indexRX),'d')
    figure(3),clf,hold on,plot(Capa,R1(:,indexRX),'<')
    figure(4),clf,hold on,plot(Capa,C1(:,indexRX),'>')
    figure(5),clf,hold on,plot(Capa,R2(:,indexRX),'*')
    figure(6),clf,hold on,plot(Capa,C2(:,indexRX),'+')
end