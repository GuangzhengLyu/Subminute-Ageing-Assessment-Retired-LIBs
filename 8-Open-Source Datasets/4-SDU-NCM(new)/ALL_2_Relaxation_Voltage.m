close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Extract and visualize relaxation-voltage segments (Vrlx)
%%% for each cell, then plot the relaxation trajectories using a capacity-
%%% ranked color gradient.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for IndexData = 1:length(OneCycle)

    % Identify relaxation indices:
    % - Current equals zero (rest / relaxation period)
    % - Voltage above a high-voltage threshold (near end of charge)
    Index1 = find(OneCycle(IndexData).CurrentA == 0);
    Index2 = find(OneCycle(IndexData).VoltageV >= 4.1);
    IndexRX = intersect(Index1,Index2);

    % Extract the relaxation-voltage segment:
    % - Start from one point before the first relaxation index (to include
    %   the transition into relaxation), and end at the last relaxation index
    Vrlx{IndexData,1} = OneCycle(IndexData).VoltageV(IndexRX(1)-1:IndexRX(end));
end

NumData = length(OneCycle);

% Collect initial capacity for sorting / ranking
for i = 1:NumData
    MyCapa(i) = OneCycle(i).OrigCapaAh;
end

% Sort by initial capacity (ascending)
[A, B] = sort(MyCapa); %#ok<ASGLU>

for i = 1:NumData
    Index = B(i);

    % =========================
    % Color mapping for visualization
    % - Early ageing: color gradient driven by capacity rank
    % =========================
    ColorMap = [ ...
        0.8*(NumData-i)/NumData, ...
        0.5 + 0.8*(NumData-i)/(2*NumData), ...
        0 ...
        ];

    % =========================
    % Alternative: color mapping for deep ageing (disabled)
    % =========================
    % if min(OneCycle(Index).Cycle.DiscCapaAh)<=2.1
    %     Life = min(find(OneCycle(Index).Cycle.DiscCapaAh <= 2.1));
    % else
    %     Life = length(OneCycle(Index).Cycle.DiscCapaAh);
    % end
    % ColorMap = [1 0.4+0.6*(Life-200)/605 0.2-0.2*(Life-200)/605];

    % Alias for readability
    RV{Index,1} = Vrlx{Index,1};

    % Plot relaxation-voltage segment
    figure(1), hold on, box on, plot(RV{Index,1},'-','color',ColorMap);
end