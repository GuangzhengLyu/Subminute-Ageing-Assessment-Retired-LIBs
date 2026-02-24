clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 3-SDU-NCM
%%% This script: Perform grouped K-means clustering using grouped, weighted
%%% Euclidean distance on (i) estimation outputs, (ii) selected health targets,
%%% (iii) relaxation features, and (iv) expanded performance indicators.
%%% Then quantify within-cluster trajectory dispersion via per-cycle standard
%%% deviation of multiple engineering signals (capacity, energy, charge capacity,
%%% charge time, and platform discharge capacity). Finally, summarize the
%%% dispersion ratios across different feature-group combinations and plot the
%%% normalized results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add grouped K-means utilities (k-means++ init, grouped distance, z-score norm)
addpath('./G-K-means')

%% ========== 1. Load grouped data ==========
load('./OneCycle_SDU_NCM.mat')  
load('./Feature_ALL_SDU_NCM.mat')
load('./PLSR_Result_1_50_Y_Test_14_SDU_NCM.mat')

% Average predictions across repeats: Estimation (N × 6)
Estimation = squeeze(mean(Y_Test,1))';

% Assemble feature tensor (sample × relaxation-segment × feature-dimension)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Convert to health-indicator groups (different "Kind" uses different groups)
% Xg{1}: capacity SOH estimate (N×1)
% Xg{2}: life/RUL estimate (N×1 here; structure preserved for extension)
% Xg{3}: relaxation feature vector (N×6)
% Xg{4}: expanded performance-indicator estimates (N×4)
Xg = cell(4,1);

for Kind = 1:4

    % Fix random seed for reproducibility (affects k-means++ init)
    rng(42);

    % Group 1: always include capacity estimate
    Xg{1} = Estimation(:,1);   % N×1

    % Group 2: optionally include life estimate
    Xg{2} = [Estimation(:,2)];   % N×1
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end

    % Group 3: optionally include relaxation features
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end

    % Group 4: optionally include expanded performance-indicator estimates
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end

    %% Parameter setup
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims); %#ok<NASGU>
    K = 20;

    %% ========== 2. Preprocessing: group-wise z-score normalization ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end

    % Concatenate groups into a single feature matrix for K-means (N × D)
    % Manual concatenation is used to avoid dimension issues in cell2mat
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};

        % Force to 2D matrix
        if isvector(A)
            A = A(:); % enforce column vector (N×1)
        end

        % If accidentally shaped as d×N, transpose to N×d
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

    %% ========== 3. Grouped K-means with weighted distance ==========
    % Group weights (set to equal weights here; keep structure for tuning)
    w = [1, 1, 1, 1];

    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple restarts, keep the best SSE
    opts.verbose = true;

    % Custom K-means: k-means++ initialization + Lloyd iterations
    % Distance: grouped weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);

    fprintf('\nClustering completed: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);

    %% ========== 4. Visualization 1: PCA 2D scatter (for plotting only) ==========
    % PCA is used only for visualization; clustering is done in the full space
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);

    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering (performed in full space; PCA only for visualization)');

    %% ========== 5. Visualization 2: cluster-center profiles by group ==========
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
        ylabel('Center value (normalized)');
        title(sprintf('Group %d (%d dims) cluster-center profiles', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end

    %% ========== 6. Basic engineering checks ==========
    % Samples per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Samples per cluster:');
    disp(countPerCluster');

    % Intra-cluster distance (weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));

    %% ========== 7. Trajectory-dispersion metrics within each cluster ==========
    % For each cluster, compute per-cycle standard deviation across samples,
    % then average over the common available cycle range.
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

        % Use the shortest trajectory length in the cluster for aligned comparison
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

        % Mean dispersion within each cluster (average over cycle index m)
        M_STD_DiscCapa(i) = mean(STD_DiscCapa{i});
        M_STD_DiscEngy(i) = mean(STD_DiscEngy{i});
        M_STD_CoChCapa(i) = mean(STD_CoChCapa{i});
        M_STD_CharTime(i) = mean(STD_CharTime{i});
        M_STD_PlfDCapa(i) = mean(STD_PlfDCapa{i});

        clear Temp
    end

    % Aggregate dispersion across clusters (one scalar per Kind)
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    % Clear large temporary containers
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables indexed by j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables indexed by m
end

%% Summary ratios and visualization
close all
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalize each metric by the baseline (Kind = 1)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Bar plots for normalized dispersion metrics
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))