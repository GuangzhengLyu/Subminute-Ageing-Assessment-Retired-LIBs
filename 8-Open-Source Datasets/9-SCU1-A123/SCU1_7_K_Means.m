clear; 
clc; 
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 9-SCU1-A123
%%% This script: Run grouped K-means clustering with group-wise z-score
%%% normalization and grouped weighted distance, then evaluate intra-cluster
%%% trajectory dispersion using cycle-by-cycle standard deviation statistics
%%% (capacity/energy/charge metrics/time/platform capacity) and summarize
%%% relative dispersion reductions across feature-group configurations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add grouped K-means utilities (k-means++ + Lloyd + grouped distance)
addpath('./G-K-means')

%% ========== 1. Load grouped data (precomputed features and predictions) ==========
load('./OneCycle_A123.mat')  
load('./Feature_ALL_A123.mat')
load('./PLSR_Result_1_80_Y_Test_14_SCUA123.mat')

% Use the mean prediction across repeats as the estimated health indicators
Estimation = squeeze(mean(Y_Test,1))';

% Stack relaxation features into a 3-D tensor: (sample, segment, feature)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Unify as health-indicator groups (for grouped K-means inputs)
% Xg{1}: capacity estimate
% Xg{2}: life estimate
% Xg{3}: relaxation-feature vector (segment 1)
% Xg{4}: extended health indicators (e.g., energy rate, CC charge rate, etc.)
Xg = cell(4,1);

for Kind = 1:4
    % Fix RNG seed for reproducibility (kept inside loop as in original script)
    rng(42);

    % Group 1: capacity estimate (always enabled)
    Xg{1} = Estimation(:,1);   % N×1

    % Group 2: life estimate (enabled for Kind = 2/3/4)
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end

    % Group 3: relaxation features (enabled for Kind = 3/4)
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end

    % Group 4: extended health indicators (enabled for Kind = 4)
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end

    %% Define clustering parameters
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims); %#ok<NASGU>
    K = 3;

    %% ========== 2. Preprocessing: group-wise z-score normalization ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end

    % Concatenate the normalized groups into one feature matrix for K-means
    % (manual concatenation to avoid dimension errors from cell2mat)
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};

        % Force 1-D vectors to be N×1
        if isvector(A)
            A = A(:);
        end

        % If accidentally shaped as d×N, transpose to N×d
        if size(A,1) ~= N && size(A,2) == N
            A = A.';
        end

        % Dimension check: must be N×d_g
        if size(A,1) ~= N
            error('Group %d has invalid size: expected %d×%d, but got %d×%d', ...
                g, N, size(A,2), size(A,1), size(A,2));
        end

        Xn = [Xn, A]; %#ok<AGROW>
    end

    %% ========== 3. Grouped K-means (group-weighted squared Euclidean distance) ==========
    % Group weights: adjust to emphasize specific groups if needed
    w = [1, 1, 1, 1];

    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple initializations, keep the best
    opts.verbose = true;

    % Custom K-means: k-means++ initialization + Lloyd iterations
    % Distance: grouped weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);

    fprintf('\nClustering completed: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);

    %% ========== 4. Visualization 1: PCA to 2D scatter (for display only) ==========
    % PCA is used only for visualization; clustering is performed in the full space
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);

    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering result (clustered in full space; PCA for visualization only)');

    %% ========== 5. Visualization 2: cluster-center profiles per group ==========
    % Split the centers back into groups and plot the profile for each group
    figure('Name','Cluster-center profiles per feature group');
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
        ylabel('Center value (after normalization)');
        title(sprintf('Group %d (%d dims) cluster-center profile', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end

    %% ========== 6. Basic engineering checks ==========
    % Sample count per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Sample count per cluster:');
    disp(countPerCluster');

    % Simple intra-cluster distance diagnostic (group-weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));

    %% ========== 7. Intra-cluster trajectory dispersion evaluation ==========
    % Compute cycle-by-cycle standard deviation within each cluster, then average
    % across cycles and clusters as a dispersion score (lower is more consistent).
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
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    % Clear temporary variables associated with cluster indexing (j) and cycle indexing (m)
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables tied to index j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables tied to index m
end

% Relative dispersion ratios: Kind=4 compared with Kind=1 (baseline)
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalize dispersion scores by baseline (Kind=1) for compact comparison
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Bar plots of relative dispersion scores across Kind = 1..4
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))