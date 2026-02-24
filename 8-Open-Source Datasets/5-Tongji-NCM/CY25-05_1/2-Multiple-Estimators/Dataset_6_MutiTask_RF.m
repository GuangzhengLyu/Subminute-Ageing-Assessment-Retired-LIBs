clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 5-Tongji-NCM (TongjiNCM25)
%%% This script: Assemble RC-type relaxation features and expanded health
%%% indicators, apply min-max normalization to inputs/outputs, and run repeated
%%% leave-one-out multi-task random-forest regression (TreeBagger) to predict
%%% multiple ageing/health targets (one forest per output). Predictions are
%%% clipped in the normalized space, de-normalized for visualization, and
%%% repeated test predictions (Y_Test) are saved to a MAT-file for downstream
%%% analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_TongjiNCM25.mat')  
load('../Feature_ALL_TongjiNCM25.mat')

% Assemble feature tensor: (sample, segment, feature-dimension)
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

    % Define life as the full available discharge-capacity trajectory length
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators (engineering-oriented performance metrics)
    ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert to normalized health indicators (relative to new-cell references)
Capa = Capa/3.5;
Life = Life;
ERate = ERate/0.93;
CoChRate = CoChRate/0.86;
MindVolt = (MindVolt-2.65)/(3.50-2.65);
PlatfCapa = PlatfCapa/1.34;

%% Output normalization (min-max scaling to [0, 1])
Max_Out = [0.925, 500, 0.98, 0.948, 0.956, 1.005];
Min_Out = [0.895, 30,  0.96, 0.936, 0.936, 0.965];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0, 1] after normalization
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14 
    %% Feature normalization (min-max scaling to [0, 1])
    Max_In = [4.164, 0.076, 0.1,   92, 0.064, 90];
    Min_In = [4.157, 0.056, 0.055, 78, 0.044, 30];

    % Map CountSV to the corresponding normalization row index (kept as original)
    MyInd = 15-CountSV;
          
    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));
    
    % Clip inputs/outputs to [0, 1] after normalization
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;
    
    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Leave-one-out split: use current sample as test, remaining as train
            Temp_F = Feature_Nor;
            Temp_O = Output;
            In_Test(:,IndexData) = Temp_F(IndexData,:)';
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';
        
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];
        
            In_Train = Temp_F';
            Out_Train = Temp_O';
            
            % Keep explicit reshaping as in original script
            In_Train = reshape(In_Train,[size(In_Train,1),size(In_Train,2)]);
            In_Test(:,IndexData)  = reshape(In_Test(:,IndexData),[size(In_Test(:,IndexData),1),size(In_Test(:,IndexData),2)]);
            
            %% Random forest regression (TreeBagger)
            trees = 100;                                      % Number of trees
            leaf  = 5;                                        % Minimum leaf size
            OOBPrediction = 'on';                             % Enable OOB error estimates
            OOBPredictorImportance = 'on';                    % Enable feature importance

            Method = 'regression';                            % Regression mode
            net_1 = TreeBagger(trees, In_Train', Out_Train(1,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_2 = TreeBagger(trees, In_Train', Out_Train(2,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_3 = TreeBagger(trees, In_Train', Out_Train(3,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_4 = TreeBagger(trees, In_Train', Out_Train(4,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_5 = TreeBagger(trees, In_Train', Out_Train(5,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
            net_6 = TreeBagger(trees, In_Train', Out_Train(6,:)', 'OOBPredictorImportance', OOBPredictorImportance,...
                  'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);

            % Predict on training set and leave-one-out test sample
            Y_Train(1,:) = predict(net_1,In_Train');
            Y_Test(CountRP,1,IndexData) = predict(net_1,In_Test(:,IndexData)');
            Y_Train(2,:) = predict(net_2,In_Train');
            Y_Test(CountRP,2,IndexData) = predict(net_2,In_Test(:,IndexData)');
            Y_Train(3,:) = predict(net_3,In_Train');
            Y_Test(CountRP,3,IndexData) = predict(net_3,In_Test(:,IndexData)');
            Y_Train(4,:) = predict(net_4,In_Train');
            Y_Test(CountRP,4,IndexData) = predict(net_4,In_Test(:,IndexData)');
            Y_Train(5,:) = predict(net_5,In_Train');
            Y_Test(CountRP,5,IndexData) = predict(net_5,In_Test(:,IndexData)');
            Y_Train(6,:) = predict(net_6,In_Train');
            Y_Test(CountRP,6,IndexData) = predict(net_6,In_Test(:,IndexData)');
        
            toc
        end

        % Clip predictions to [0, 1] in the normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;
        
        % De-normalization (back to physical/engineering units)
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
        
        % Result visualization: train fit and leave-one-out test predictions
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')
            
            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    % Save repeated prediction tensor for this setting
    currentFile = sprintf('Result_6_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end