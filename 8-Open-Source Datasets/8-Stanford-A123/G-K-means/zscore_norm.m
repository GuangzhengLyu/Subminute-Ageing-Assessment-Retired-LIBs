function [Xn, stat] = zscore_norm(X)
% zscore_norm - 稳健版z-score标准化：不改变输入矩阵形状
% 支持：X为N×d 或 N×1 向量
% 输出Xn与X同尺寸

% 确保是二维矩阵（避免某些情况下被当成行向量）
if isvector(X)
    X = X(:); % 强制列向量
end

mu = mean(X, 1);
sd = std(X, 0, 1);
sd(sd < 1e-12) = 1;

Xn = (X - mu) ./ sd;

stat.mu = mu;
stat.sd = sd;
end
