function [labelsBest, CBest, info] = kmeans_pp_lloyd(X, K, groupDims, w, opts)
% kmeans_pp_lloyd - 完整K-means：k-means++ 初始化 + Lloyd迭代
% 兼容 4/5 个输入参数：
%   (X,K,groupDims,w) 或 (X,K,groupDims,w,opts)

% ====== 兼容：如果只传了4个输入，就补默认opts ======
if nargin < 5 || isempty(opts)
    opts = struct();
end

% ---- 参数默认值 ----
if ~isfield(opts,'maxIter'),  opts.maxIter = 200; end
if ~isfield(opts,'tol'),      opts.tol = 1e-6; end
if ~isfield(opts,'nInit'),    opts.nInit = 5; end
if ~isfield(opts,'verbose'),  opts.verbose = false; end

[N, D] = size(X);
if sum(groupDims) ~= D
    error('groupDims维度之和必须等于特征总维度D。');
end
if numel(w) ~= numel(groupDims)
    error('权重w长度必须等于组数。');
end

bestSSE = inf;
labelsBest = ones(N,1);
CBest = zeros(K, D);
bestIter = 0;

for trial = 1:opts.nInit
    C = init_kmeanspp(X, K, groupDims, w);

    prevSSE = inf;
    labels = ones(N,1);

    for it = 1:opts.maxIter
        dist2 = grouped_distance_sq(X, C, groupDims, w); % N×K
        [mind2, labels] = min(dist2, [], 2);
        SSE = sum(mind2);

        Cnew = C;
        for k = 1:K
            idxk = (labels == k);
            if any(idxk)
                Cnew(k,:) = mean(X(idxk,:), 1);
            else
                % 空簇补救：随机挑一个点
                Cnew(k,:) = X(randi(N), :);
            end
        end

        if abs(prevSSE - SSE) <= opts.tol * max(1, prevSSE)
            C = Cnew;
            break;
        end

        C = Cnew;
        prevSSE = SSE;
    end

    if opts.verbose
        fprintf('Init %d/%d: SSE=%.4f, iter=%d\n', trial, opts.nInit, SSE, it);
    end

    if SSE < bestSSE
        bestSSE = SSE;
        labelsBest = labels;
        CBest = C;
        bestIter = it;
    end
end

info.bestSSE = bestSSE;
info.bestIter = bestIter;

end

function C = init_kmeanspp(X, K, groupDims, w)
% k-means++ 初始化
[N, D] = size(X);
C = zeros(K, D);

C(1,:) = X(randi(N), :);

for k = 2:K
    dist2 = grouped_distance_sq(X, C(1:k-1,:), groupDims, w);
    dmin = min(dist2, [], 2);

    p = dmin / sum(dmin + eps);
    cdf = cumsum(p);
    r = rand();
    idx = find(cdf >= r, 1, 'first');
    if isempty(idx), idx = randi(N); end

    C(k,:) = X(idx,:);
end
end
