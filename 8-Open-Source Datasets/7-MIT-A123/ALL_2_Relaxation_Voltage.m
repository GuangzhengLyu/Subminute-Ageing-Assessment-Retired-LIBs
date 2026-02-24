%%% ========================================================================
%%% Project : External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset : 7-MIT-A123
%%% This script: Extract and visualize relaxation-voltage segments (Vrlx)
%%%             sorted by initial capacity, with a rank-based colormap.
%%% ========================================================================

close all

% -------------------------------------------------------------------------
% 1) Extract relaxation-voltage segment (Vrlx) for each cell/sample
% -------------------------------------------------------------------------
for IndexData = 1:length(OneCycle)

    % Find indices where current is exactly zero (rest/relaxation)
    Index1 = find(OneCycle(IndexData).CurrentA == 0);

    % Keep only the last continuous block of zero-current indices
    IndexRX = Index1;
    TempIR = IndexRX;
    for i = 1:length(IndexRX)-1
        if IndexRX(i+1)-IndexRX(i) > 1
            TempIR = IndexRX(i+1:end);
        end
    end
    IndexRX = TempIR;

    % Extract relaxation voltage segment (include one sample right before rest)
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

% -------------------------------------------------------------------------
% 2) Sort samples by initial capacity and plot relaxation-voltage curves
% -------------------------------------------------------------------------
NumData = length(OneCycle);

for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end
[A B] = sort(MyCapa);

for i = 1:NumData
    Index = B(i);

    % Rank-based colormap for early ageing visualization
    ColorMap = [0.8*(NumData-i)/NumData 0.5+0.8*(NumData-i)/(2*NumData) 0];

    % Alternative colormap for deep ageing visualization (life-based)
    % Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % ColorMap = [1 0.4+0.6*(Life-190)/2000 0.2-0.2*(Life-190)/2000];

    % Relaxation voltage curve
    RV{Index,1} = Vrlx{Index,1};

    % Plot
    figure(1),hold on,box on,plot(RV{Index,1},'-','color',ColorMap);
end