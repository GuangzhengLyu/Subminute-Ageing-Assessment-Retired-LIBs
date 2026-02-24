clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: Tongji-NCA (TongjiNCA25)
%%% This script: Evaluate sorting/grouping consistency using grouped K-means.
%%% It builds grouped feature sets from PLSR estimations and relaxation
%%% features, performs weighted grouped K-means clustering, and quantifies
%%% within-cluster trajectory dispersion (STD over cycle index) for multiple
%%% raw trajectories. Results are summarized as relative dispersion ratios
%%% across feature-group configurations and saved for downstream reporting.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./G-K-means')

%% Data loading (grouped inputs)
% Load structured single-cycle dataset, extracted relaxation features, and
% repeated LOOCV PLSR predictions (used here as estimation outputs).
load('./OneCycle_TongjiNCA25.mat')  
load('./Feature_ALL_TongjiNCA25.mat')
load('./PLSR_Result_1_70_Y_Test_14_TongjiNCA25.mat')

% Use the mean prediction across repetitions as the point estimation (N x 6)
Estimation = squeeze(mean(Y_Test,1))';

%% Feature tensor assembly
% Feature(:,:,k) stores the k-th relaxation-derived parameter across
% samples (row) and voltage setpoints (column). For this dataset, the
% feature matrices are provided as [N x 1] and are placed into Feature(:,1,k).
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Grouped feature configuration
% Xg{g} stores the g-th feature group. "Kind" controls which groups are
% activated (others are set to zeros to isolate their effect).
Xg = cell(4,1);

for Kind = 1:4
    rng(42); % Fixed random seed for reproducibility

    % Group 1: capacity-based SOH estimate (N x 1)
    Xg{1} = Estimation(:,1);   % N×1
    
    % Group 2: life estimate (N x 1)
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end
    
    % Group 3: relaxation feature vector (N x 6) at setpoint index 1
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end
    
    % Group 4: expanded health-indicator estimates (N x 4)
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Parameter setup
    % N: sample number; groupDims: feature dimensions per group; K: clusters
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims);
    K = 5;
    
    %% Preprocessing: group-wise standardization (recommended)
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    %% Concatenate normalized groups into a single matrix
    % Manual concatenation is used to avoid cell2mat dimension errors.
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Force 2-D shape (N x d_g)
        if isvector(A)
            A = A(:); % enforce column vector (N×1)
        end
    
        % If mistakenly shaped as d_g x N, transpose to N x d_g
        if size(A,1) ~= N && size(A,2) == N
            A = A.';
        end
    
        % Dimension check: must be N x d_g
        if size(A,1) ~= N
            error('Group %d dimension mismatch: expected %d×%d, but got %d×%d.', ...
                g, N, size(A,2), size(A,1), size(A,2));
        end
    
        Xn = [Xn, A]; %#ok<AGROW>
    end
    
    %% Grouped weighted K-means (group-weighted distance)
    % You can set per-group weights w to reflect group importance.
    % Example: w = [1, 1, 2, 0.8] upweights group 3 (6-D relaxation features).
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple initializations, keep the best
    opts.verbose = true;
    
    % Custom K-means: k-means++ initialization + Lloyd iterations
    % Distance: grouped weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering completed: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% Visualization 1: PCA projection (2D)
    % PCA is used only for visualization; clustering is performed in D-D space.
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering results (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering results (clustering in high-D; PCA for visualization only)');
    
    %% Visualization 2: cluster-center profiles by group
    % Split C back into groups and plot center values (in normalized space).
    figure('Name','Cluster-center profiles by feature group');
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
        title(sprintf('Group %d (%d-D): cluster-center profile', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% Engineering checks
    % Samples per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Samples per cluster:');
    disp(countPerCluster');
    
    % Mean within-cluster distance (weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean within-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% Within-cluster trajectory dispersion (STD over cycle index)
    % For each cluster, compute the STD across samples at each cycle index,
    % then average over the common cycle range (min trajectory length).
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

    % Aggregate mean dispersion across clusters (per Kind configuration)
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    % Clear cluster-level buffers before the next Kind iteration
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables indexed by j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables indexed by m
end

% Relative dispersion ratios (Kind=4 vs Kind=1 baseline)
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

%% Result summary and visualization
% Normalize dispersion summaries by the baseline (Kind=1) and plot bar charts.
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))

% Save normalized dispersion results for downstream reporting
save K_Means_Result_25.mat Result