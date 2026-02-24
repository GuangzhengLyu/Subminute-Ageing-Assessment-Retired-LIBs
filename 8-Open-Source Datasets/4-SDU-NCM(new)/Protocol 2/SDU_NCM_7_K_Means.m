clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 4-SDU-NCM(new)
%%% This script: Perform grouped-feature K-means clustering (with optional
%%% group activation), then evaluate intra-cluster trajectory dispersion
%%% using standard deviation profiles of multiple engineering indicators
%%% (capacity/energy/charge metrics/time/platform capacity). Results are
%%% aggregated into normalized ratios and saved for downstream comparison.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add grouped K-means utilities (k-means++ init + grouped weighted distance)
addpath('./G-K-means')

%% ========== 1. Load grouped data ==========
load('./OneCycle_SDU_NCM_P2.mat')  
load('./Feature_ALL_SDU_NCM_P2.mat')
load('./PLSR_Result_1_80_Y_Test_14_SDU_NCM_P2.mat')

% Use the mean prediction over repeated runs as the estimation result (N×6)
Estimation = squeeze(mean(Y_Test,1))';

% Assemble relaxation features into a 3D tensor for consistent access
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Unify as health indicators (grouped inputs for clustering)
% Xg is a cell array of feature groups:
%   Group 1: capacity-related estimate (N×1)
%   Group 2: life-related estimate (N×1)          (enabled for Kind >= 2)
%   Group 3: relaxation-parameter features (N×6)  (enabled for Kind >= 3)
%   Group 4: expanded indicators (N×4)            (enabled for Kind == 4)
Xg = cell(4,1);

for Kind = 1:4

    % Fix random seed for reproducibility
    rng(42); % fixed random seed for reproducibility

    Xg{1} = Estimation(:,1);   % N×1
    
    Xg{2} = [Estimation(:,2)];   % N×2
    Xg{2} = 0*ones(size(Xg{2},1),size(Xg{2},2));
    if ismember(Kind,[2,3,4])
        Xg{2} = [Estimation(:,2)];
    end
    
    Xg{3} = squeeze(Feature(:,1,:));   % N×6
    Xg{3} = 0*ones(size(Xg{3},1),size(Xg{3},2));
    if ismember(Kind,[3,4])
        Xg{3} = squeeze(Feature(:,1,:));
    end
    
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Define clustering parameters
    N = size(Xg{1},1);
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims);
    K = 3;
    
    %% ========== 2. Pre-processing: group-wise standardization ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    % Concatenate all groups back into a single feature matrix for K-means
    % ===== Manual concatenation to avoid cell2mat dimension issues =====
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Enforce 2D matrix form
        if isvector(A)
            A = A(:); % force column vector (N×1)
        end
    
        % If mistakenly converted to d×N, transpose back to N×d
        if size(A,1) ~= N && size(A,2) == N
            A = A.';
        end
    
        % Dimension check: must be N×d_g
        if size(A,1) ~= N
            error('Group %d has wrong size: expected %d×%d, but got %d×%d', ...
                g, N, size(A,2), size(A,1), size(A,2));
        end
    
        Xn = [Xn, A]; %#ok<AGROW>
    end
    
    %% ========== 3. K-means with grouped weighted distance ==========
    % Set group weights to reflect relative importance of each feature group
    % Example: emphasize group 3 (6D) by setting w = [1, 1, 2, 0.8]
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple initializations, keep the best
    opts.verbose = true;
    
    % Custom K-means (k-means++ initialization + Lloyd iterations)
    % Distance: grouped weighted squared Euclidean distance
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering finished: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% ========== 4. Visualization 1: PCA to 2D scatter ==========
    % PCA is only used for visualization; clustering is performed in D-D space
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering (performed in D-D, PCA is for visualization only)');
    
    %% ========== 5. Visualization 2: cluster-center profiles per group ==========
    % Split the learned center matrix C back into groups and plot group-wise profiles
    figure('Name','Group-wise cluster-center profiles');
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
        title(sprintf('Group %d (%dD) center profiles', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% ========== 6. Engineering checks ==========
    % Sample counts per cluster
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Sample counts per cluster:');
    disp(countPerCluster');
    
    % Simple evaluation: average intra-cluster distance (weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% ========== 7. Compute intra-cluster trajectory dispersion ==========
    % Compute STD profiles over cycle index for each cluster and each indicator
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

    % Aggregate dispersion metrics across clusters for this Kind
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables related to index j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables related to index m
end

close all

% Report relative improvements (Kind 4 vs Kind 1) for selected metrics
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalize results by the baseline (Kind 1)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Bar plots for normalized dispersion ratios
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))

% Save summary result vector
save K_Means_Result_P2.mat Result