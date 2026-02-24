clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 10-Stanford-LMO
%%% This script: Build normalized health-indicator targets and normalized
%%% relaxation-feature inputs, then run repeated leave-one-out multi-task
%%% SVR (RBF kernel) to predict six ageing indicators. For each output, the
%%% script performs a grid search for (C, gamma) using cross-validation and
%%% then predicts train/test outputs. Predicted test outputs (Y_Test) are
%%% saved to MAT-files for downstream evaluation and result aggregation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_Stanford_LMO.mat')  
load('../Feature_ALL_Stanford_LMO.mat')

% Add SVR library functions (LIBSVM)
addpath('./4-SVR');

% Assemble feature tensor (kept mapping and ordering as in original script)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: trajectory length of discharge-capacity series (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Extended health indicators (dataset-provided scalar metrics)
    ERate(CountData,1)     = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1)  = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1)  = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to unified health indicators (simple scaling to comparable ranges)
Capa = Capa/32.5;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.7-3);
PlatfCapa = PlatfCapa/20;

%% Target normalization (0–1) using fixed min/max bounds
Max_Out = [0.55, 2100,  0.97,  0.9998, 0.96, 0.6];
Min_Out = [0.25, 1200,  0.926, 0.9995, 0.76, 0.15];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));

% Extended health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip normalized targets to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

% Loop over the specified setting index (kept original loop bounds)
for CountSV = 14:-1:14 
    %% Input feature normalization (0–1) using fixed min/max bounds
    Max_In = [3.93, 0.0065, 0.00115, 1800000, 0.001,  4000000];
    Min_In = [3.87, 0.003,  0.0007,  200000,  0.0004, 0];

    % Select which bound set to use (kept original mapping)
    MyInd = 15-CountSV;
          
    % Normalize each relaxation feature dimension
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip normalized inputs and targets to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    % Repeated runs (kept original repetition count)
    for CountRP = 1:2
        % Leave-one-out evaluation over all samples
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create per-iteration copies for leave-one-out split
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Hold out one sample as test
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            % Remove the held-out sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            % Arrange data as (features x samples) and (targets x samples)
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Preserve original reshape operations (explicit dimensional intent)
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% IV. SVR model creation/training (RBF kernel)
            %% Model 1
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(1,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_1 = svmtrain(Out_Train(1,:)',In_Train',cmd);

            %% Model 2
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(2,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_2 = svmtrain(Out_Train(2,:)',In_Train',cmd);

            %% Model 3
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(3,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_3 = svmtrain(Out_Train(3,:)',In_Train',cmd);

            %% Model 4
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(4,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_4 = svmtrain(Out_Train(4,:)',In_Train',cmd);

            %% Model 5
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(5,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_5 = svmtrain(Out_Train(5,:)',In_Train',cmd);

            %% Model 6
            % 1. Grid search for best C and gamma
            [c,g] = meshgrid(-15:0.5:15,-15:0.5:15);
            [m,n] = size(c);
            cg = zeros(m,n);  
            eps = 10^(-4);
            v = 2;
            bestc = 0;
            bestg = 0;
            error = Inf;
            for i = 1:m
                for j = 1:n
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV error; -s 3 for SVR regression
                    cg(i,j) = svmtrain(Out_Train(6,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% best C
                        bestg = 2^g(i,j);% best gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% prioritize smaller C if tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% build training command with best C and gamma
            model_6 = svmtrain(Out_Train(6,:)',In_Train',cmd);

            % V. SVR prediction
            Y_Train(1,:) = svmpredict(Out_Train(1,:)',In_Train',model_1);
            Y_Test(CountRP,1,IndexData) = svmpredict(Out_Test(CountRP,1,IndexData)' ,In_Test(:,IndexData)',model_1);

            Y_Train(2,:) = svmpredict(Out_Train(2,:)',In_Train',model_2);
            Y_Test(CountRP,2,IndexData) = svmpredict(Out_Test(CountRP,2,IndexData)' ,In_Test(:,IndexData)',model_2);

            Y_Train(3,:) = svmpredict(Out_Train(3,:)',In_Train',model_3);
            Y_Test(CountRP,3,IndexData) = svmpredict(Out_Test(CountRP,3,IndexData)' ,In_Test(:,IndexData)',model_3);

            Y_Train(4,:) = svmpredict(Out_Train(4,:)',In_Train',model_4);
            Y_Test(CountRP,4,IndexData) = svmpredict(Out_Test(CountRP,4,IndexData)' ,In_Test(:,IndexData)',model_4);

            Y_Train(5,:) = svmpredict(Out_Train(5,:)',In_Train',model_5);
            Y_Test(CountRP,5,IndexData) = svmpredict(Out_Test(CountRP,5,IndexData)' ,In_Test(:,IndexData)',model_5);

            Y_Train(6,:) = svmpredict(Out_Train(6,:)',In_Train',model_6);
            Y_Test(CountRP,6,IndexData) = svmpredict(Out_Test(CountRP,6,IndexData)' ,In_Test(:,IndexData)',model_6);
        
            toc
        end

        % Clip predicted normalized outputs to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        % De-normalize: convert predictions and ground truth back to original scales
        Y_Train(1,:) = (Y_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Train(1,:) = (Out_Train(1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Y_Train(2,:) = Y_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        Out_Train(2,:) = Out_Train(2,:)*(Max_Out(2)-Min_Out(2))+Min_Out(2);
        
        Y_Test(CountRP,1,:) = (Y_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        Out_Test(CountRP,1,:) = (Out_Test(CountRP,1,:)*(Max_Out(1)-Min_Out(1))+Min_Out(1));
        for IndOp = 2:6
            Y_Test(CountRP,IndOp,:) = Y_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
            Out_Test(CountRP,IndOp,:) = Out_Test(CountRP,IndOp,:)*(Max_Out(IndOp)-Min_Out(IndOp))+Min_Out(IndOp);
        end
        
        % Visualization: scatter plots (ground truth vs prediction) for train/test
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save predicted test outputs for this setting
    currentFile = sprintf('Result_4_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end