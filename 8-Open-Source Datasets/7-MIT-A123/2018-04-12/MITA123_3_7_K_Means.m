clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123 (MITA123_3)
%%% This script: Perform grouped K-means clustering with group-weighted
%%% distances on PLSR estimations and relaxation features, then quantify
%%% intra-cluster trajectory dispersion (STD-based) and summarize results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add grouped K-means utilities to path
addpath('./G-K-means')

%% ========== 1. Load data (grouped inputs) ==========
load('./OneCycle_MITA123_3.mat')  
load('./Feature_ALL_MITA123_3.mat')
load('./PLSR_Result_1_80_Y_Test_14_MITA123_3.mat')

% Use the mean prediction across repeated runs as the estimation output
Estimation = squeeze(mean(Y_Test,1))';

% Pack relaxation features into a 3-D tensor for consistent indexing
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Convert to health-indicator groups
% Xg is a cell array of feature groups; each group can be toggled on/off via Kind
Xg = cell(4,1);

for Kind = 1:4
    rng(42); % Fix random seed for reproducibility

    % Group 1: capacity estimation (N×1)
    Xg{1} = Estimation(:,1);   % N×1
    
    % Group 2: life estimation (N×1; optionally enabled)
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end
    
    % Group 3: relaxation features (N×6; optionally enabled)
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end
    
    % Group 4: extended health indicators (N×4; optionally enabled)
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Set parameters
    N = size(Xg{1},1);  
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)]; 
    G = numel(groupDims);
    D = sum(groupDims);
    K = 8;
    
    %% ========== 2. Pre-processing: group-wise standardization ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    % Concatenate all groups into a single feature matrix for K-means
    % Manual concatenation is used to avoid cell2mat dimension issues.
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Force to a 2-D matrix (N×d_g)
        if isvector(A)
            A = A(:); % Force column vector (N×1)
        end
    
        % If converted to d×N by mistake, transpose to N×d
        if size(A,1) ~= N && size(A,2) == N
            A = A.';
        end
    
        % Dimension check: must be N×d_g
        if size(A,1) ~= N
            error('Group %d dimension mismatch: expected %d×%d, but got %d×%d', ...
                g, N, size(A,2), size(A,1), size(A,2));
        end
    
        Xn = [Xn, A]; %#ok<AGROW>
    end
    
    %% ========== 3. K-means with grouped weighted distance ==========
    % Group weights reflect relative importance across feature groups.
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % Multiple initializations, keep the best
    opts.verbose = true;
    
    % Custom K-means (k-means++ initialization + Lloyd iterations)
    % Distance: grouped weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering finished: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% ========== 4. Visualization 1: PCA projection to 2D ==========
    % PCA is used for visualization only (clustering is done in the full space).
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering result (clustered in full space; PCA for visualization)');
    
    %% ========== 5. Visualization 2: cluster-center profiles per group ==========
    % Split centers C back into groups and plot the per-group center patterns.
    figure('Name','Cluster-center profiles by group');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    
    startCol = 1;
    for g = 1:G
        d = groupDims(g);
        cols = startCol:(startCol+d-1);
        startCol = startCol + d;
    
        nexttile;
        plot(1:d, C(:, cols)', '-o', 'LineWidth', 1.2);
        grid on;
        xlabel(sprintf('Group %d feature index', g));
        ylabel('Center value (standardized)');
        title(sprintf('Group %d (%d-D) cluster-center profile', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% ========== 6. Basic cluster sanity checks ==========
    % Sample count per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Sample count per cluster:');
    disp(countPerCluster');
    
    % Intra-cluster mean distance (weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% ========== 7. Intra-cluster trajectory dispersion (STD-based) ==========
    % Compute per-cycle dispersion within each cluster for multiple trajectories,
    % then aggregate to obtain a mean STD score per cluster and per Kind.
    for i = 1:K
        IndexLable = find(labels == i);
        for j = 1:length(IndexLable)
            DiscCapa{i}{j} = OneCycle(IndexLable(j)).Cycle.DiscCapaAh;
            DiscEngy{i}{j} = OneCycle(IndexLable(j)).Cycle.DiscEnergyWh;
            CoChCapa{i}{j} = OneCycle(IndexLable(j)).Cycle.ConstCharCapaAh;
            T = OneCycle(IndexLable(j)).Cycle.CharTimeS;
            T = T(isfinite(T));
            CharTime{i}{j} = T;
            PlfDCapa{i}{j} = OneCycle(IndexLable(j)).Cycle.PlatfCapaAh;
            Length{i}(j) = length(PlfDCapa{i}{j});
        end
        for m = 1:min(Length{i})
            for j = 1:length(IndexLable)
                Temp(j) = DiscCapa{i}{j}(m);
            end
            STD_DiscCapa{i}(m) = std(Temp);

            for j = 1:length(IndexLable)
                Temp(j) = DiscEngy{i}{j}(m);
            end
            STD_DiscEngy{i}(m) = std(Temp);

            for j = 1:length(IndexLable)
                Temp(j) = CoChCapa{i}{j}(m);
            end
            STD_CoChCapa{i}(m) = std(Temp);

            for j = 1:length(IndexLable)
                Temp(j) = CharTime{i}{j}(m);
            end
            STD_CharTime{i}(m) = std(Temp);

            for j = 1:length(IndexLable)
                Temp(j) = PlfDCapa{i}{j}(m);
            end
            STD_PlfDCapa{i}(m) = std(Temp);
        end
        M_STD_DiscCapa(i) = mean(STD_DiscCapa{i});
        M_STD_DiscEngy(i) = mean(STD_DiscEngy{i});
        M_STD_CoChCapa(i) = mean(STD_CoChCapa{i});
        M_STD_CharTime(i) = mean(STD_CharTime{i});
        M_STD_PlfDCapa(i) = mean(STD_PlfDCapa{i});

        clear Temp
    end

    % Aggregate STD metrics across clusters (one value per Kind)
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    % Clear loop variables (indexed by j / m) to avoid carry-over between Kinds
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables indexed by j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables indexed by m
end

% Relative improvement ratios (Kind 4 vs Kind 1) for selected metrics
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalize results by Kind 1 (baseline)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Bar plots for each dispersion metric across Kind settings
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))

% Save aggregated results
save K_Means_Result_3.mat Result
