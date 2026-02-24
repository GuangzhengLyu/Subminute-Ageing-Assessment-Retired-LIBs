clear; 
clc; 
close all;
rng(42); % 固定随机种子，保证可复现

%% ========== 1. 生成一个"分组结构相似"的数值数据集 ==========
N = 240;                      
groupDims = [1 2 6 4];        
G = numel(groupDims);
D = sum(groupDims);         

K = 4; 

% 生成每一类在每一组的"均值中心"（让不同组对分类贡献不同）
% 你也可以把这些中心理解成"工程上不同健康状态/工况的多维指标模式"
mu = zeros(K_true, D);
idx = [0, cumsum(groupDims)];

for k = 1:K_true
    % 第1组（1维）：区分度中等
    mu(k, idx(1)+1:idx(2)) = 1.5*(k-1);

    % 第2组（2维）：区分度较强
    mu(k, idx(2)+1:idx(3)) = [2.0*(k-1), -1.2*(k-1)];

    % 第3组（6维）：区分度最强（模拟"信息量最大的一组"）
    base = (k-1)*1.0;
    mu(k, idx(3)+1:idx(4)) = base + linspace(-1, 1, groupDims(3));

    % 第4组（4维）：区分度较弱（模拟"噪声更大/更不稳定的一组"）
    mu(k, idx(4)+1:idx(5)) = 0.6*(k-1) + [0.2 -0.1 0.1 -0.2];
end

% 为每个类分配样本
label_true = randi(K_true, N, 1);

% 每组噪声水平（可调）：第4组噪声更大
sigma_group = [0.7, 0.5, 0.35, 1.0];

X = zeros(N, D);
for i = 1:N
    k = label_true(i);
    for g = 1:G
        cols = (idx(g)+1):idx(g+1);
        X(i, cols) = mu(k, cols) + sigma_group(g)*randn(1, numel(cols));
    end
end

% 构造"分组特征"cell，工程上常用这种数据组织方式
Xg = cell(G,1);
for g = 1:G
    cols = (idx(g)+1):idx(g+1);
    Xg{g} = X(:, cols);
end

%% ========== 2. 预处理：按组标准化（推荐） ==========
Xg_norm = cell(G,1);
for g = 1:G
    [Xg_norm{g}, ~] = zscore_norm(Xg{g});
end

% 拼回总特征矩阵（给K-means用）
% ===== 手动拼接：避免cell2mat在维度不一致时直接报错 =====
Xn = [];
for g = 1:numel(Xg_norm)
    A = Xg_norm{g};

    % 统一成二维矩阵
    if isvector(A)
        A = A(:); % 强制列向量（N×1）
    end

    % 如果误变成 d×N，则转置成 N×d
    if size(A,1) ~= N && size(A,2) == N
        A = A.';
    end

    % 维度检查：必须是 N×d_g
    if size(A,1) ~= N
        error('第%d组维度错误：期望 %d×%d，但实际是 %d×%d', ...
            g, N, size(A,2), size(A,1), size(A,2));
    end

    Xn = [Xn, A]; %#ok<AGROW>
end

%% ========== 3. K-means：支持"分组加权距离" ==========
% 你可以设置每组权重，体现"某些组更重要"
% 例如让第3组(6维)权重大一些：w = [1, 1, 2, 0.8]
w = [1, 1, 2, 0.8];

opts.maxIter = 200;
opts.tol = 1e-6;
opts.nInit = 8;         % 多次初始化取最优
opts.verbose = true;

% 用我们自己写的K-means（k-means++初始化 + Lloyd）
% 距离用"分组加权欧氏距离平方"
[labels, C, info] = kmeans_pp_lloyd(Xn, K, groupDims, w, opts);

fprintf('\n完成聚类：best SSE = %.4f, iter = %d\n', info.bestSSE, info.bestIter);

%% ========== 4. 可视化 1：PCA 降维到2D散点 ==========
% PCA只是为了画图，不参与聚类（聚类在13维空间完成）
[coeff, score] = pca(Xn); %#ok<ASGLU>
Z = score(:,1:2);

figure('Name','K-means聚类结果（PCA 2D 可视化）');
gscatter(Z(:,1), Z(:,2), labels);
grid on; xlabel('PC1'); ylabel('PC2');
title('K-means聚类结果（在13维上聚类，PCA仅用于可视化）');

%% ========== 5. 可视化 2：每组"簇中心"的特征剖面 ==========
% 将中心C拆回到每一组，画每组中心的均值曲线
figure('Name','各组特征簇中心剖面');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

startCol = 1;
for g = 1:G
    d = groupDims(g);
    cols = startCol:(startCol+d-1);
    startCol = startCol + d;

    nexttile;
    plot(1:d, C(:, cols)', '-o', 'LineWidth', 1.2);
    grid on;
    xlabel(sprintf('组%d特征维度索引', g));
    ylabel('中心值（标准化后）');
    title(sprintf('组%d（%d维）簇中心剖面', g, d));
    legend(arrayfun(@(k)sprintf('簇%d',k), 1:K, 'UniformOutput', false), ...
           'Location','best');
end

%% ========== 6. 输出一些工程常用检查信息 ==========
% 各簇样本数
countPerCluster = accumarray(labels, 1, [K 1]);
disp('各簇样本数：');
disp(countPerCluster');

% 简单评估：簇内平均距离（加权）
dist2 = grouped_distance_sq(Xn, C, groupDims, w); % N×K
intra = sqrt(dist2(sub2ind(size(dist2), (1:N)', labels)));
fprintf('簇内平均距离(加权) = %.4f\n', mean(intra));
