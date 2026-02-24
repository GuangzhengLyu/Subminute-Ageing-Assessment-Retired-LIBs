clear
clc
% 示例数据
X = [1, 2;
     3, 4;
     5, 6];   % n × p 自变量矩阵
X1 = [2, 4;
     6, 8;
     10, 12];   % n × p 自变量矩阵
y = [1; 2; 3];           % n × 1 因变量

% 多元线性回归
mdl = fitlm(X, y);

% 查看结果
disp(mdl)

% 回归系数
beta = mdl.Coefficients.Estimate;

% 预测
y_pred = predict(mdl, X1);