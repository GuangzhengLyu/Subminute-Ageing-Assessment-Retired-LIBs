clear; 
clc; 
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 10-Stanford-LMO
%%% This script: Run grouped/weighted K-means clustering on health-related
%%% estimations and relaxation features, then quantify within-cluster
%%% trajectory dispersion using engineering indicators (capacity, energy,
%%% charge time, etc.). Results are summarized as normalized dispersion
%%% ratios and visualized with bar charts.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add custom grouped K-means utilities (k-means++ + Lloyd with grouped distance)
addpath('./G-K-means')

%% ========== 1. Load grouped data ==========
load('./OneCycle_Stanford_LMO.mat')  
load('./Feature_ALL_Stanford_LMO.mat')
load('./PLSR_Result_1_50_Y_Test_14_Stanford_LMO.mat')

% Aggregate repeated predictions: mean over repetitions, then reshape to N x outputs
Estimation = squeeze(mean(Y_Test,1))';

% Assemble relaxation feature tensor (kept mapping and ordering as in original script)
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Convert to unified health indicators (grouped feature sets)
% Xg contains 4 groups that can be toggled on/off via Kind
Xg = cell(4,1);

for Kind = 1:4

    % Fix random seed for reproducibility (k-means initialization)
    rng(42); % fixed random seed to ensure reproducibility

    % Group 1: primary scalar indicator
    Xg{1} = Estimation(:,1);   % N×1
    
    % Group 2: optional scalar indicator (enabled for Kind = 2/3/4)
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
    
    % Group 4: extended estimated indicators (enabled for Kind = 4)
    Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];   % N×4
    Xg{4} = 0*ones(size(Xg{4},1),size(Xg{4},2));
    if ismember(Kind,[4])
        Xg{4} = [Estimation(:,3),Estimation(:,4),Estimation(:,5),Estimation(:,6)];
    end
    
    %% Define parameters
    N = size(Xg{1},1); 
    groupDims = [size(Xg{1},2) size(Xg{2},2) size(Xg{3},2) size(Xg{4},2)];
    G = numel(groupDims);
    D = sum(groupDims);
    K = 5;
    
    %% ========== 2. Pre-processing: group-wise standardization ==========
    Xg_norm = cell(G,1);
    for g = 1:G
        [Xg_norm{g}, ~] = zscore_norm(Xg{g});
    end
    
    % Concatenate groups into one feature matrix for K-means
    % Manual concatenation is kept to avoid cell2mat dimension errors
    Xn = [];
    for g = 1:numel(Xg_norm)
        A = Xg_norm{g};
    
        % Ensure a 2D matrix representation
        if isvector(A)
            A = A(:); % force column vector (N×1)
        end
    
        % If accidentally transposed to d×N, transpose back to N×d
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
    % Set per-group weights (kept as equal weights in original script)
    % Example: emphasize the 3rd group (6-D) via w = [1, 1, 2, 0.8]
    w = [1, 1, 1, 1];
    
    opts.maxIter = 200;
    opts.tol = 1e-6;
    opts.nInit = 8;         % multiple initializations, keep best solution
    opts.verbose = true;
    
    % Custom K-means (k-means++ initialization + Lloyd iterations)
    % Distance: grouped weighted squared Euclidean
    [labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);
    
    fprintf('\nClustering finished: best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);
    
    %% ========== 4. Visualization 1: PCA 2D scatter ==========
    % PCA is only for plotting; clustering is performed in the full feature space
    [coeff, score] = pca(Xn); %#ok<ASGLU>
    Z = score(:,1:2);
    
    figure('Name','K-means clustering result (PCA 2D visualization)');
    gscatter(Z(:,1), Z(:,2), labels);
    grid on; xlabel('PC1'); ylabel('PC2');
    title('K-means clustering (clustering in high-D space; PCA for visualization only)');
    
    %% ========== 5. Visualization 2: group-wise cluster-center profiles ==========
    % Split cluster centers C back into groups and plot per-group center profiles
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
        title(sprintf('Group %d (%d-D) center profiles', g, d));
        legend(arrayfun(@(k)sprintf('Cluster %d',k), 1:K, 'UniformOutput', false), ...
               'Location','best');
    end
    
    %% ========== 6. Engineering checks ==========
    % Cluster sample counts
    countPerCluster = accumarray(labels, 1, [K 1]);
    disp('Samples per cluster:');
    disp(countPerCluster');
    
    % Intra-cluster mean distance (weighted)
    dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
    intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
    fprintf('Mean intra-cluster distance (weighted) = %.4f\n', mean(intra));
    
    %% ========== 7. Compute within-cluster trajectory dispersion ==========
    % Dispersion is quantified by the mean standard deviation across cycles
    % for several engineering trajectories (capacity, energy, charge time, etc.)
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

    % Aggregate dispersion across clusters for this Kind setting
    M_STD_DiCaSum(Kind) = mean(M_STD_DiscCapa);
    M_STD_DiEnSum(Kind) = mean(M_STD_DiscEngy);
    M_STD_CCCaSum(Kind) = mean(M_STD_CoChCapa);
    M_STD_ChTiSum(Kind) = mean(M_STD_CharTime);
    M_STD_PDCaSum(Kind) = mean(M_STD_PlfDCapa);

    % Clear loop-scope variables tied to indices j and m
    clear DiscCapa DiscEngy CoChCapa CharTime PlfDCapa Length % variables tied to index j
    clear STD_DiscCapa STD_DiscEngy STD_CoChCapa STD_CharTime STD_PlfDCapa % variables tied to index m
end

% Close all figures generated inside the loop
close all

% Relative dispersion ratios: Kind=4 vs Kind=1 (printed as a quick summary)
[M_STD_DiCaSum(4)/M_STD_DiCaSum(1),M_STD_DiEnSum(4)/M_STD_DiEnSum(1),M_STD_ChTiSum(4)/M_STD_ChTiSum(1),M_STD_PDCaSum(4)/M_STD_PDCaSum(1),]
close all

% Normalized results (each metric divided by the Kind=1 baseline)
Result(1,:) = M_STD_DiCaSum./M_STD_DiCaSum(1);
Result(2,:) = M_STD_DiEnSum./M_STD_DiEnSum(1);
Result(3,:) = M_STD_CCCaSum./M_STD_CCCaSum(1);
Result(4,:) = M_STD_ChTiSum./M_STD_ChTiSum(1);
Result(5,:) = M_STD_PDCaSum./M_STD_PDCaSum(1);

% Visualization: bar charts of normalized dispersion ratios
figure(1),clf,bar(Result(1,:))
figure(2),clf,bar(Result(2,:))
figure(3),clf,bar(Result(3,:))
figure(4),clf,bar(Result(4,:))
figure(5),clf,bar(Result(5,:))