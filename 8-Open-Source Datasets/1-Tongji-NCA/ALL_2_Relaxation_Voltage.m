close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA
%%% Description:
%%%   Extract relaxation-voltage segments at the end of charge for each cell,
%%%   then sort cells by BOL discharge capacity (OrigCapaAh) and plot the
%%%   relaxation-voltage trajectories with a rank-based colormap.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for IndexData = 1:length(OneCycle)

    % Identify relaxation samples (I = 0) and near-EOC high-voltage samples (V >= 4 V)
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV >= 4);
    IndexRX = intersect(Index1,Index2);

    % Extract relaxation-voltage segment (include the sample right before relaxation begins)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

NumData = length(OneCycle);

% Collect BOL capacity of each cell for sorting
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort cells by BOL capacity (ascending)
[A B] = sort(MyCapa);

for i = 1:NumData
    Index = B(i);

    %% Early ageing (colormap based on sorted BOL capacity rank)
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    %% Deep ageing (alternative colormap based on cycle life; disabled by default)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1 0.4+0.6*(Life-190)/2000 0.2-0.2*(Life-190)/2000];

    % Relaxation voltage trajectory for the selected cell
    RV{Index,1} = Vrlx{Index,1};
    
    % Plot relaxation voltage traces; color encodes cell ordering
    figure(1),hold on,box on,plot(RV{Index,1},'-','color',ColorMap);
end