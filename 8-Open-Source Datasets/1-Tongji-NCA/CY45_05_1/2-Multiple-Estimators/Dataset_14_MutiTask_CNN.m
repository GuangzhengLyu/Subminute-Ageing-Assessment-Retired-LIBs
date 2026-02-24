clear
close all
% clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% This script: Build normalized targets (capacity SOH proxy, life proxy, and
%%% four expanded health indicators) and normalized relaxation features
%%% (Uoc, R0, R1, C1, R2, C2). A repeated leave-one-out workflow is used to
%%% train and test a compact CNN-based multi-output regressor (trainNetwork)
%%% that maps a stacked feature "image" to six health indicators. Test
%%% predictions are stored in Y_Test and saved to a MAT-file for later RMSE
%%% evaluation and visualization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('../OneCycle_TongjiNCA.mat')
load('../Feature_ALL_TongjiNCA.mat')

% Pack relaxation features
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction (targets)
CountData = 0;
for IndexData = 1:length(OneCycle)
    CountData = CountData+1;

    % Life proxy (cycle count of available discharge-capacity series)
    Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);

    % Original capacity (Ah)
    Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

    % Expanded health indicators
    ERate(CountData,1)    = OneCycle(IndexData).Cycle.EnergyRate(2);
    CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
    MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
    PlatfCapa(CountData,1)= OneCycle(IndexData).Cycle.PlatfCapaAh(1);
end

%% Convert raw measurements into normalized health indicators
% NOTE: Scaling factors are kept exactly as in the original script
Capa      = Capa/3.5;
Life      = Life;
ERate     = ERate/0.93;
CoChRate  = CoChRate/0.86;
MindVolt  = (MindVolt-2.65)/(3.54-2.65);
PlatfCapa = PlatfCapa/1.35;

%% Output normalization (min–max)
Max_Out = [0.952, 800, 0.985, 0.994, 0.992, 0.986];
Min_Out = [0.943, 200, 0.976, 0.982, 0.978, 0.966];

Output(:,1) = (Capa      -Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life      -Min_Out(2))/(Max_Out(2)-Min_Out(2));
Output(:,3) = (ERate     -Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate  -Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt  -Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa -Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0, 1]
Output(Output<0) = 0;Output(Output>1) = 1;

for CountSV = 14:-1:14
    %% Feature normalization (min–max per relaxation feature)
    % NOTE: Bounds are kept exactly as in the original script
    Max_In = [4.1718, 0.054, 0.049,  102, 0.049, 102];
    Min_In = [4.1698, 0.046, 0.046,  90,  0.046, 91];

    % Index mapping kept to preserve the original structure
    MyInd = 15-CountSV;

    Feature_Nor(:,1) = (Feature(:,1,1)-Min_In(MyInd,1))/(Max_In(MyInd,1)-Min_In(MyInd,1));
    Feature_Nor(:,2) = (Feature(:,1,2)-Min_In(MyInd,2))/(Max_In(MyInd,2)-Min_In(MyInd,2));
    Feature_Nor(:,3) = (Feature(:,1,3)-Min_In(MyInd,3))/(Max_In(MyInd,3)-Min_In(MyInd,3));
    Feature_Nor(:,4) = (Feature(:,1,4)-Min_In(MyInd,4))/(Max_In(MyInd,4)-Min_In(MyInd,4));
    Feature_Nor(:,5) = (Feature(:,1,5)-Min_In(MyInd,5))/(Max_In(MyInd,5)-Min_In(MyInd,5));
    Feature_Nor(:,6) = (Feature(:,1,6)-Min_In(MyInd,6))/(Max_In(MyInd,6)-Min_In(MyInd,6));

    % Clip features to [0, 1]
    Feature_Nor(Feature_Nor<0) = 0;Feature_Nor(Feature_Nor>1) = 1;
    Output(Output<0) = 0;Output(Output>1) = 1;

    for CountRP = 1:100
        for IndexData = 1:length(Feature_Nor)
            tic
            CountRP
            IndexData

            % Leave-one-out split
            Temp_F = Feature_Nor;
            Temp_O = Output;

            % Format the held-out sample as a 4-D "image" tensor
            % (height = #stacked feature channels, width = 6, channels = 1, batch = 1)
            for NumST = 1:size(Temp_F,3)
                In_Test(NumST,:,1,1) = Temp_F(IndexData,:,NumST);
            end
            Out_Test(CountRP,:,IndexData) = Temp_O(IndexData,:)';

            % Remove held-out sample from training data
            Temp_F(IndexData,:) = [];
            Temp_O(IndexData,:) = [];

            % Stack all remaining samples into 4-D training tensor
            for NumST = 1:size(Temp_F,3)
                for NumID = 1:size(Temp_F,1)
                    In_Train(NumST,:,1,NumID) = Temp_F(NumID,:,NumST);
                end
            end
            Out_Train = Temp_O';

            %% CNN-based multi-output regressor
            % Input is treated as an "image" of size [#channels-by-6-by-1]
            layers = [
                imageInputLayer([size(In_Train,1) 6 1])

                % Convolution + pooling block
                convolution2dLayer([ceil(size(In_Train,1)/2),3],20,'Padding','same')
                maxPooling2dLayer([1,1],'Stride',2)

                % Fully connected regression head
                fullyConnectedLayer(10)
                fullyConnectedLayer(6)
                regressionLayer];

            options = trainingOptions(...
                'adam', ...
                'MaxEpochs',           200, ...
                'MiniBatchSize',       32, ...
                'InitialLearnRate',    0.01, ...
                'GradientThreshold',   0.9, ...
                'LearnRateSchedule',   'piecewise', ...
                'LearnRateDropPeriod', 5, ...
                'LearnRateDropFactor', 0.9, ...
                'Verbose',             0, ...
                'Plots',               'none'...
                );

            % Train and predict
            net = trainNetwork(In_Train,Out_Train',layers,options);

            Y_Train = predict(net,In_Train)';
            Y_Test(CountRP,:,IndexData) = predict(net,In_Test);

            toc
        end

        % Clip test predictions in normalized space
        Temp = Y_Test(CountRP,:,:);
        Temp(Temp<0) = 0;Temp(Temp>1) = 1;
        Y_Test(CountRP,:,:) = Temp;

        %% Denormalize back to original scales
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

        %% Visualization (ground truth vs prediction)
        for IndOTrain = 1:6
            figure(IndOTrain),hold on
            plot(Out_Train(IndOTrain,:),Y_Train(IndOTrain,:),'o')

            figure(IndOTrain+6),hold on
            plot(squeeze(Out_Test(CountRP,IndOTrain,:)),squeeze(Y_Test(CountRP,IndOTrain,:)),'o')
        end
    end

    currentFile = sprintf('Result_14_Y_Test_%d.mat',CountSV);
    save(currentFile,'Y_Test')
end