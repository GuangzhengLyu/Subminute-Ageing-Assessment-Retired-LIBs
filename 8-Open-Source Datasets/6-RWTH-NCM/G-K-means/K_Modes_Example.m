%% =========================================================
% kmodes_engineering_demo.m
% 完整可运行的 K-modes 工程案例（脚本）
% - 数据结构：三组特征 1/2/4（共7列，均为类别型）
% - 算法：K-modes（mismatch距离 + mode中心）
% - 可视化：One-hot+PCA二维散点、簇大小、场景分布
% MATLAB: R2025B 可运行
%% =========================================================
clc; clear; close all;
rng(1);

%% -------------------- 1) 生成"结构相似"的类别数据集 --------------------
N = 150;        % 样本数（你可改）
K = 3;          % 簇数（你可改）

% 三组特征：第1组1列；第2组2列；第3组4列
% 这里用 string 模拟（工程里你也可以用 categorical/cellstr，后面统一转 string）
% 组1：退役场景
X1 = randsample(["accident","normal","post"], N, true);

% 组2：厂商、型号
X2 = [ ...
    randsample(["CATL","BYD","EVE"], N, true), ...
    randsample(["typeA","typeB","typeC"], N, true) ...
];

% 组3：4个离散等级（可理解为"分箱后的健康特征等级"）
X3 = [ ...
    randsample(["L","M","H"], N, true), ...
    randsample(["low","mid","high"], N, true), ...
    randsample(["Q1","Q2","Q3","Q4"], N, true), ...
    randsample(["ok","warn","bad"], N, true) ...
];

X = [X1, X2, X3];     % N×7

% 注入"真实一些的簇结构"：让数据不是纯随机，聚类有意义
idxA = 1:round(N/3);
idxB = round(N/3)+1:round(2*N/3);
idxC = round(2*N/3)+1:N;

X(idxA,1) = "accident"; X(idxA,2) = "CATL"; X(idxA,4) = "H";   X(idxA,7) = "warn";
X(idxB,1) = "normal";   X(idxB,2) = "BYD";  X(idxB,4) = "M";   X(idxB,7) = "ok";
X(idxC,1) = "post";     X(idxC,2) = "EVE";  X(idxC,4) = "L";   X(idxC,7) = "bad";

% 少量噪声（工程更真实）
noiseRate = 0.08;
numNoise = round(N*noiseRate);
noiseRows = randsample(N, numNoise);
X(noiseRows, 2) = randsample(["CATL","BYD","EVE"], numNoise, true);

%% -------------------- 2) K-modes 参数 --------------------
groupIdx = {1, 2:3, 4:7};   % 1/2/4 分组
groupW   = [1, 1, 1];       % 分组权重（可改：比如第三组更重要 => [1 1 2]）

maxIter = 60;
nInit   = 10;

%% -------------------- 3) 运行严格 K-modes（多次初始化取最优） --------------------
X = string(X);
[N, D] = size(X);

bestCost = inf;
bestLabels = [];
bestModes  = [];

for init = 1:nInit
    % 随机初始化中心：取K个样本
    initIdx = randperm(N, K);
    modes = X(initIdx, :);
    labels = ones(N,1);

    for it = 1:maxIter
        oldLabels = labels;

        % ---- 分配：计算每个样本到每个中心的分组加权 mismatch 代价 ----
        costMat = zeros(N, K);
        for k = 1:K
            mk = modes(k,:);
            c  = zeros(N,1);
            for g = 1:numel(groupIdx)
                cols = groupIdx{g};
                mismatch = sum(X(:,cols) ~= mk(:,cols), 2);
                c = c + groupW(g) * mismatch;
            end
            costMat(:,k) = c;
        end
        [minCost, labels] = min(costMat, [], 2);

        % ---- 更新：每个簇逐列取众数(mode)作为新中心 ----
        for k = 1:K
            members = (labels == k);
            if ~any(members)
                % 空簇：重置
                modes(k,:) = X(randi(N), :);
            else
                Xk = X(members,:);
                for j = 1:D
                    col = Xk(:,j);
                    [u,~,ic] = unique(col);
                    cnt = accumarray(ic,1);
                    [~, im] = max(cnt);
                    modes(k,j) = u(im);
                end
            end
        end

        % 收敛判定
        if isequal(labels, oldLabels)
            break;
        end
    end

    totalCost = sum(minCost);
    if totalCost < bestCost
        bestCost   = totalCost;
        bestLabels = labels;
        bestModes  = modes;
    end
end

fprintf("K-modes finished. Best cost = %.0f\n", bestCost);

%% -------------------- 4) 可视化：One-hot + PCA 到 2D 散点 --------------------
% 将每列类别做 one-hot，然后 PCA 到二维
Z = onehot_encode_string_matrix(X);   % N×M 数值矩阵
Z = zscore(Z);                        % 标准化（让PCA更稳定）
[coeff, score] = pca(Z);              % score: N×M
XY = score(:,1:2);

figure;
gscatter(XY(:,1), XY(:,2), bestLabels);
grid on;
xlabel("PC1");
ylabel("PC2");
title("K-modes clustering (One-hot + PCA 2D projection)");

%% -------------------- 5) 可视化：每簇样本数 --------------------
clusterCounts = zeros(K,1);
for k = 1:K
    clusterCounts(k) = sum(bestLabels==k);
end

figure;
bar(clusterCounts);
grid on;
xlabel("Cluster");
ylabel("Count");
title("Cluster sizes (K-modes)");

%% -------------------- 6) 可视化：第1组特征(场景)在各簇分布 --------------------
scenes = ["accident","normal","post"];
sceneCountMat = zeros(K, numel(scenes));

for k = 1:K
    Xk = X(bestLabels==k, 1); % 场景列
    for s = 1:numel(scenes)
        sceneCountMat(k,s) = sum(Xk == scenes(s));
    end
end

figure;
bar(sceneCountMat, 'stacked');
grid on;
xlabel("Cluster");
ylabel("Count");
legend(scenes, "Location","best");
title("Group-1 feature distribution across clusters (retirement scenario)");

%% -------------------- 7) 打印中心（工程报告友好） --------------------
disp("==============================================");
disp("Mode centers (K-modes):");
for k = 1:K
    fprintf("Cluster %d center:\n", k);
    disp(bestModes(k,:));
end
disp("==============================================");

%% =========================================================
% 本脚本用到的"脚本内工具函数"（不需要额外文件）
%% =========================================================
function Z = onehot_encode_string_matrix(X)
% X: N×D string
% Z: N×M double
% 对每一列单独 unique，并做 one-hot 拼接

X = string(X);
[N, D] = size(X);

Z = [];
for j = 1:D
    col = X(:,j);
    cats = unique(col);         % 该列所有类别
    Mj = numel(cats);

    Zj = zeros(N, Mj);
    for c = 1:Mj
        Zj(:,c) = (col == cats(c));
    end

    Z = [Z, Zj]; %#ok<AGROW>
end
end
