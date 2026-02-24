clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Perform grouped, weighted K-means clustering on different
%%% combinations of (i) model estimations, (ii) cycle-life estimation,
%%% (iii) relaxation-feature vectors, and (iv) expanded health indicators.
%%% The clustering result is then evaluated by intra-cluster trajectory
%%% dispersion (STD over cycle-wise trajectories) for multiple engineering
%%% signals, and the normalized dispersion ratios are saved and visualized.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add grouped K-means utilities (custom implementation)
addpath('./G-K-means')

%% ========== 1. Load data (grouped inputs) ==========
load('./OneCycle_SDU_NCM_P1.mat')  
load('./Feature_ALL_SDU_NCM_P1.mat')
load('./PLSR_Result_1_80_Y_Test_14_SDU_NCM_P1.mat')

% Use the mean prediction over repeated runs as the point estimate (N×6)
Estimation = squeeze(mean(Y_Test,1))';

% Pack fitted relaxation parameters into a 3-D feature tensor
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Normalize to health-indicator inputs (grouped feature design)
% Xg is a grouped cell array:
%   Group 1: capacity estimation (N×1)
%   Group 2: life estimation (N×1, enabled for Kind = 2/3/4)
%   Group 3: relaxation features (N×6, enabled for Kind = 3/4)
%   Group 4: expanded health indicators (N×4, enabled for Kind = 4)
Xg = cell(4,1);

for Kind = 1:4

    % Fix random seed for reproducibility
    rng(42);

    % Group 1: capacity-related estimation
    Xg{1} = Estimation(:,1);   % N×1
    
    % Group 2: life-related estimation (optional by Kind)
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end
    
    % Group 3: relaxation features (optional by Kind)
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end
    
    % Group 4: expanded health indicators (optional by Kind)
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Determine parameters
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims);
    K = 3;
    
    %% ========== 2. Pre-processing: group-wise normalization (recommended) ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    % Concatenate into the full feature matrix for K-means
    % ===== Manual concatenation to avoid dimension mismatch errors =====
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Force into a 2-D matrix
        if isvector(A)
            A = A(:); % enforce column vector (N×1)
        end
    
        % If mistakenly shaped as d×N, transpose to N×d
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
    % Group weights (all groups equally weighted by default)
    % Example: emphasize Group 3 (6-D) via w = [1, 1, 2, 0.8]
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple initializations, take the best
    opts.verbose = true;
    
    % Custom K-means: k-means++ initialization + Lloyd iterations
    % Distance: grouped, weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering finished: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% ========== 4. Visualization 1: PCA projection to 2D scatter ==========
    % PCA is only for plotting; clustering is performed in the full feature space
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering result (clustered in full space; PCA for visualization only)');
    
    %% ========== 5. Visualization 2: cluster-center profiles per group ==========
    % Split the full center matrix C back into groups and plot each group profile
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
        ylabel('Center value (z-score normalized)');
        title(sprintf('Group %d (%d-D) cluster-center profile', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% ========== 6. Engineering sanity checks ==========
    % Number of samples per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Samples per cluster:');
    disp(countPerCluster');
    
    % Intra-cluster mean distance (group-weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% ========== 7. Compute intra-cluster trajectory dispersion (STD) ==========
    % For each cluster, compute cycle-wise STD across samples for:
    %   - Discharge capacity
    %   - Discharge energy
    %   - Constant-current charge capacity
    %   - Charge time (finite entries only)
    %   - Platform discharge capacity
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

    % Clear Kind-specific temporary variables (indexed by j / m)
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables indexed by j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables indexed by m
end
close all

% Ratio check: Kind=4 versus Kind=1 (kept as in original script)
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalize dispersion metrics by the baseline (Kind=1)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Bar plots: normalized dispersion ratios across Kind = 1..4
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))

% Save clustering-evaluation results
save K_Means_Result_P1.mat Result