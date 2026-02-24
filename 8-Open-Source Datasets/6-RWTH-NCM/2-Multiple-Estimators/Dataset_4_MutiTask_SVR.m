clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Build multi-task targets (capacity, life, and expanded
%%% performance indicators), normalize inputs/outputs, and run repeated
%%% leave-one-out Support Vector Regression (SVR, RBF kernel) using RC
%%% relaxation features as inputs. Hyperparameters (C and gamma) are tuned
%%% by grid-search cross-validation for each target. Predictions are
%%% de-normalized for visualization and saved to MAT files.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_NCM_RWTH.mat')  
load('../Feature_ALL_RWTH_NCM.mat')

% SVR library functions (LIBSVM-style)
addpath('./4-SVR');

% Assemble feature tensor: each slice is one relaxation feature matrix
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Build per-sample ground-truth targets from the cycling trajectories
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy: full trajectory length (cycles)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Unify as health-indicator targets
% Apply dataset-specific scaling to map each indicator into a comparable range
Capa = Capa/2;
Life = Life;
ERate = ERate/0.99;
CoChRate = CoChRate/1;
MindVolt = (MindVolt-3)/(3.9-3);
PlatfCapa = PlatfCapa/1.5;

%% Target normalization (min-max)
% NOTE: Max_Out/Min_Out are fixed normalization bounds used for all samples
Max_Out = [0.925, 1800, 1.012, 0.46, 0.946, 0.344];
Min_Out = [0.89,  1200, 1.002, 0.36, 0.934, 0.326];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clamp normalized targets to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (min-max)
    % NOTE: Max_In/Min_In are fixed normalization bounds for the input features.
    % The indexing pattern (MyInd = 15-CountSV) is preserved from the original script.
    Max_In = [4.095, 2.1,  0.4,  1600, 0.17, 900];
    Min_In = [4.082, 1.96, 0.04, 0,    0.03, 0];

    MyInd = 15-CountSV;

    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clamp normalized features/targets to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    %% Repeated leave-one-out SVR (RBF kernel) with per-task grid search
    for CountRP = 1:2
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Create per-iteration train/test split (leave-one-out)
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove test sample from training set
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Training matrices (features: rows, samples: columns)
            In_Train = Temp_F';
            Out_Train = Temp_O';

            % Preserve original reshape operations
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);

            %% IV. SVM model creation/training (RBF kernel)
            %% Model 1
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(1,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
            model_1 = svmtrain(Out_Train(1,:)',In_Train',cmd);

            %% Model 2
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(2,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
            model_2 = svmtrain(Out_Train(2,:)',In_Train',cmd);

            %% Model 3
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(3,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
            model_3 = svmtrain(Out_Train(3,:)',In_Train',cmd);

            %% Model 4
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(4,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
            model_4 = svmtrain(Out_Train(4,:)',In_Train',cmd);

            %% Model 5
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(5,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
            model_5 = svmtrain(Out_Train(5,:)',In_Train',cmd);

            %% Model 6
            % 1. Search best C / gamma (grid-search CV)
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
                    cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 Gaussian kernel; -v returns CV score; -s 3 for SVR
                    cg(i,j) = svmtrain(Out_Train(6,:)',In_Train',cmd);
                    if cg(i,j) < error
                        error = cg(i,j);
                        bestc = 2^c(i,j);% C
                        bestg = 2^g(i,j);% gamma
                    end
                    if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)% Prefer smaller C when tied
                        error = cg(i,j);
                        bestc = 2^c(i,j);
                        bestg = 2^g(i,j);
                    end
                end
            end

            % 2. Train SVR with best C and gamma
            cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];% Use best C and gamma
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

        % Clamp normalized predictions to [0, 1]
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% De-normalization (inverse min-max)
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

        %% Visualization (train vs test scatter)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save predictions for this CountSV setting
    currentFile = sprintf('Result_4_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end