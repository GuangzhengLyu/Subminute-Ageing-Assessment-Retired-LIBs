%% 决策树回归最简版本
clear; clc;

% 1. 数据
N = 50; M = 3; L = 2;
X = randn(N, M);
y(:,1) = 2*X(:,1) + randn(N,1)*0.1;  % 输出1
y(:,2) = 3*X(:,2) + randn(N,1)*0.1; % 输出2

% 2. 训练决策树（每个输出一个树）
trees = cell(L, 1);
y_pred = zeros(N, L);

for i = 1:L
    trees{i} = fitrtree(X, y(:, i));
    y_pred(:, i) = predict(trees{i}, X);
end

% 3. 简单显示
figure;
for i = 1:L
    subplot(1, L, i);
    plot(y(:, i), y_pred(:, i), 'o');
    hold on;
    plot(xlim, xlim, 'r-');
    xlabel('真实值'); ylabel('预测值');
    title(['输出 ', num2str(i)]);
end