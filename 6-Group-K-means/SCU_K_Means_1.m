clear; 
clc; 
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Sorting Accuracy Evaluation via Group-Weighted K-means
%%% This script: Perform grouped, weighted K-means clustering on SCU3 Dataset
%%% #1 using different feature-group combinations (Kind = 1..4), then quantify
%%% within-cluster trajectory dispersion via standard deviations of capacity/
%%% energy/charge/time/platform-capacity sequences and summarize results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ========== 1. Load grouped data ==========
% SCU3 Dataset #1
% Load structured single-cycle dataset, relaxation-derived feature tensors,
% and saved PLSR predictions (used as estimated health indicators).
load('../OneCycle_1.mat')  
load('../Feature_1_ALL.mat')
load('../1-Proposed/Result/PLSR_Result_1_70_Y_Test_13.mat')

% Add custom grouped K-means utilities to path
addpath('Group_K_Means')

% Use the mean prediction across repetitions as the point estimate (N x 6)
Estimation = squeeze(mean(Y_Test,1))';

%% Feature tensor assembly (relaxation parameters at multiple setpoints)
% Feature(:,:,k) stores the k-th relaxation-derived parameter across samples
% (row) and voltage setpoints (column).
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Unified indicator groups (Xg) for grouped clustering
% Xg is a 4-group cell array. Each Kind selects an increasing set of groups:
%   Kind = 1: group-1 only
%   Kind = 2: groups 1-2
%   Kind = 3: groups 1-3
%   Kind = 4: groups 1-4
Xg = cell(4,1);

for Kind = 1:4
    rng(42); % Fix the random seed for reproducible initialization
    
    % Group 1: capacity-based SOH estimate (N x 1)
    Xg{1} = Estimation(:,1);   % N×1
    
    % Group 2: life / RUL-related estimate (N x 1)
    % Default to zeros, then enable for selected Kind values
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end
    
    % Group 3: relaxation-feature vector at setpoint index 13 (N x 6)
    % Default to zeros, then enable for selected Kind values
    Xg{3} = squeeze(Feature(:,13,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,13,:));
    end
    
    % Group 4: expanded performance-indicator estimates (N x 4)
    % Default to zeros, then enable for Kind = 4
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Parameter setup
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims);
    K = 25; % 15,20,25
    
    %% ========== 2. Pre-processing: per-group z-score normalization ==========
    % Standardize each group independently to balance scales before applying
    % grouped weighted distance in K-means.
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    %% Concatenate normalized groups into a single feature matrix Xn (N x D)
    % Manual concatenation avoids cell2mat issues when shapes are inconsistent.
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Enforce a 2-D matrix representation
        if isvector(A)
            A = A(:); % Force column vector (N x 1)
        end
    
        % If accidentally shaped as d x N, transpose to N x d
        if size(A,1) ~= N && size(A,2) == N
            A = A.';
        end
    
        % Dimension check: must be N x d_g
        if size(A,1) ~= N
            error('Group %d dimension mismatch: expected %d×%d, but got %d×%d', ...
                g, N, size(A,2), size(A,1), size(A,2));
        end
    
        Xn = [Xn, A]; %#ok<AGROW>
    end
    
    %% ========== 3. Group-weighted K-means clustering ==========
    % Set per-group weights (can be tuned to emphasize certain groups).
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % Multiple initializations, select best SSE
    opts.verbose = true;
    
    % Custom K-means implementation:
    %   - k-means++ initialization
    %   - Lloyd iterations
    %   - distance: grouped weighted squared Euclidean
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering finished: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% ========== 4. Visualization 1: PCA 2D scatter ==========
    % PCA is for visualization only; clustering is performed in the full space.
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering (clustered in full space; PCA only for plotting)');
    
    %% ========== 5. Visualization 2: Cluster-center profiles per group ==========
    % Split cluster centers C back into groups and plot each group's center profile.
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
        ylabel('Center value (normalized)');
        title(sprintf('Group %d (%d dims): cluster-center profiles', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% ========== 6. Basic checks ==========
    % Cluster population counts
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Samples per cluster:');
    disp(countPerCluster');
    
    % Mean within-cluster distance under grouped weighted metric
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean within-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% ========== 7. Within-cluster trajectory dispersion (STD over cycles) ==========
    % For each cluster:
    %   - gather cycle trajectories (capacity/energy/charge/time/platform-capacity)
    %   - compute per-cycle STD across samples up to the minimum available length
    %   - average STD over cycles, then average over clusters for each Kind
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

    % Clear per-cluster buffers (loop-index dependent variables)
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables tied to index j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables tied to index m
end

%% Summary ratios (Kind = 4 relative to Kind = 1)
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

%% Result normalization (relative to Kind = 1 baseline)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

%% Bar plots for each dispersion metric across Kind = 1..4
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))