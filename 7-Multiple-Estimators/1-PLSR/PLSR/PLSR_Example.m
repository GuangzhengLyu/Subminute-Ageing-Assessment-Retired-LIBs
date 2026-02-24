%% PLSR最简版本
clear; clc;

% 1. 数据
N = 50; M = 8; L = 2;
X = randn(N, M);
y(:,1) = 2*X(:,1) + randn(N,1)*0.1;  % 输出1
y(:,2) = 3*X(:,2) + randn(N,1)*0.1; % 输出2

% 2. 训练PLSR（自动选择成分数）
ncomp = min(5, M);  % 最多5个成分
[~, ~, ~, ~, beta] = plsregress(X, y, ncomp);

% 3. 预测
y_pred = [ones(N,1), X] * beta;

% 4. 简单显示
figure;
for i = 1:L
    subplot(1, L, i);
    plot(y(:,i), y_pred(:,i), 'o');
    hold on;
    plot(xlim, xlim, 'r-');
    xlabel('真实值'); ylabel('预测值');
    title(['输出 ', num2str(i)]);
end