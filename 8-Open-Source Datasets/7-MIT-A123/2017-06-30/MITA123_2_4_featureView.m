clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123
%%% This script: Process the OneCycle_MITA123_2 dataset, fit relaxation
%%% traces, assemble feature matrices (Uoc, R0, R1, C1, R2, C2), and save
%%% them for downstream ageing assessment workflows.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset
load("OneCycle_MITA123_2.mat")

tic
CountData = 0;
CountFit = 0;

for IndexData = 1:length(OneCycle)
    figure(7),clf
    CountData = CountData+1;

    % Define life as the first cycle where discharge capacity drops below 0.88 Ah
    % If the threshold is not reached, use the full trajectory length
    if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 0.88)
        Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 0.88));
    else
        Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
    end

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

   %% Relaxation-voltage extraction
    % Initialize outputs for this sample (kept as explicit zero assignment)
    Uoc(CountData,1) = 0;
    R0(CountData,1) = 0;
    R1(CountData,1) = 0;
    C1(CountData,1) = 0;
    R2(CountData,1) = 0;
    C2(CountData,1) = 0;

    % Sample inclusion/exclusion gate (kept as in original script)
    if ~ismember(IndexData,[])
        CountFit = CountFit+1;
        CountRX = 0;
        MRMSE = 0;

        % Relaxation segment index (CountRX is designed for multiple segments;
        % here it is used once, but the structure is preserved)
        CountRX = CountRX+1;
        LengthDrop = 0;
        LengthCops = 0;

        % Fixed terminal-voltage setpoint for relaxation extraction
        Vset = 2.0;

        % Identify relaxation points: zero-current region
        Index1 = find(OneCycle(IndexData).CurrentA == 0);
        IndexRX = Index1;

        % Keep the last continuous zero-current block (preserve original logic)
        TempIR = IndexRX;
        for i = 1:length(IndexRX)-1
            if IndexRX(i+1)-IndexRX(i) > 1
                TempIR = IndexRX(i+1:end);
            end
        end
        IndexRX = TempIR;

        % Store relaxation voltage trace (include one point before the first zero-current index)
        Vrlx{CountData,1} = OneCycle(IndexData).VoltageV(IndexRX(1+LengthDrop)-1:IndexRX(end));

        % Record the start/end indices of the relaxation segment in the raw series
        PointRX(CountRX,1) = IndexRX(1);
        PointRX(CountRX,2) = IndexRX(end);

      %% Health-feature construction
        % Use the relaxation voltage trace as the fitting target
        MyData = Vrlx{CountData,1};

        % Record the ending relaxation voltage (for quick sanity checks/plots)
        EndValueRX(CountData,CountRX) = MyData(end);

        % Construct an implicit time index (unit sample index, offset by LengthCops)
        TimeInP = LengthCops+1:LengthCops+length(MyData);

        % Manually tuned bias adjustment for specific samples (kept as in original)
        if ismember(IndexData, [])
            Bias = 0.1;
        elseif ismember(IndexData, [])
            Bias = 0.2;
        elseif ismember(IndexData, [])
            Bias = 0.3;
        elseif ismember(IndexData, [])
            Bias = 0.4;
        elseif ismember(IndexData, [])
            Bias = 0.5;
        elseif ismember(IndexData, [])
            Bias = 0.6;
        else
            Bias = -0.4;
        end

        % Initial parameter guess: [A1, tau1, A2, tau2, offset]
        beta0=[0.1 10 0.1 10 Vset-Bias];

        % Two-term exponential relaxation model with offset
        mymodel = @(beta,x) beta(1)*exp(-x./beta(2))+...
                            beta(3)*exp(-x./beta(4))+...
                            beta(5);

        % Nonlinear least-squares fit and prediction
        [Beta(CountData,:),r,J]= nlinfit(TimeInP',MyData,mymodel,beta0);
        [Y,delta]=nlpredci(mymodel,TimeInP,Beta(CountData,:),r,J);

        % Fit error metrics: RMSE for this relaxation segment and running mean (MRMSE)
        RMSE = sqrt(sum((MyData-Y).^2)/length(MyData));
        MRMSE = (MRMSE*(CountRX-1)+RMSE)/CountRX;
        MMRMSE_Fit(CountFit) = MRMSE;

        % Visualization: measured vs fitted relaxation, plotted relative to Vset
        figure(7),hold on,box on
        plot([MyData-Vset],'linewidth', 2,'color',[0 CountRX/13 1-CountRX/26])
        plot([Y-Vset],'-','linewidth', 2,'color',[0 0 0])
        title('MRMSE=', MRMSE)

        % Enforce a consistent parameter ordering: ensure Beta(:,1) >= Beta(:,3)
        % (swap the two exponential terms if needed)
        if Beta(CountData,1) < Beta(CountData,3)
            Temp13 = Beta(CountData,1);
            Temp24 = Beta(CountData,2);
            Beta(CountData,1) = Beta(CountData,3);
            Beta(CountData,2) = Beta(CountData,4);
            Beta(CountData,3) = Temp13;
            Beta(CountData,4) = Temp24;
        end

      %% Feature construction
        % Convert fitted parameters into RC-type relaxation features
        CutCurrent = 0.025;

        Uoc(CountData,CountRX) = Beta(CountData,5);
        R0(CountData,CountRX)  = (Vset-Beta(CountData,1)-Beta(CountData,3)-Beta(CountData,5))/CutCurrent;
        R1(CountData,CountRX)  = abs(Beta(CountData,1)/CutCurrent);
        C1(CountData,CountRX)  = abs((Beta(CountData,2)+LengthDrop)/R1(CountData,CountRX));
        R2(CountData,CountRX)  = abs(Beta(CountData,3)/CutCurrent);
        C2(CountData,CountRX)  = abs((Beta(CountData,4)+LengthDrop)/R2(CountData,CountRX));
    end
end
toc

% Report mean fitting error across all fitted samples
mean(MMRMSE_Fit)

% Quick check: original capacity vs relaxation ending voltage
figure(1),hold on,plot(Capa,EndValueRX(:,1),'o')

% Scatter plots: original capacity vs each fitted relaxation feature
for i = 1:-1:1
    indexRX = i;

    figure(1),clf,hold on,plot(Capa,Uoc(:,indexRX),'o')
    figure(2),clf,hold on,plot(Capa,R0(:,indexRX),'d')
    figure(3),clf,hold on,plot(Capa,R1(:,indexRX),'<')
    figure(4),clf,hold on,plot(Capa,C1(:,indexRX),'>')
    figure(5),clf,hold on,plot(Capa,R2(:,indexRX),'*')
    figure(6),clf,hold on,plot(Capa,C2(:,indexRX),'+')
end

% Save assembled relaxation-feature matrices for downstream modelling
save Feature_ALL_MITA123_2.mat Uoc R0 R1 C1 R2 C2