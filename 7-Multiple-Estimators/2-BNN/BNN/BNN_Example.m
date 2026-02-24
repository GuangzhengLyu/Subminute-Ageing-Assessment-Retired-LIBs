%% 多输出贝叶斯神经网络
clear; clc;

% 1. 生成多输出数据
X = randn(80, 2);  % 80×2输入
y1 = sin(X(:,1)) + randn(80,1)*0.1;  % 输出1
y2 = cos(X(:,2)) + randn(80,1)*0.1;  % 输出2
y = [y1, y2];  % 80×2输出

% 2. 创建贝叶斯正则化神经网络
net = fitnet([10, 8], 'trainbr');  % 两个隐藏层：10和8个神经元
net.trainParam.showWindow = false;
net.trainParam.epochs = 100;

% 3. 训练
net = train(net, X', y');

% 4. 预测
y_pred = net(X')';

% 5. 显示两个输出的结果
figure;
for i = 1:2
    subplot(1,2,i);
    scatter(y(:,i), y_pred(:,i), 'filled');
    hold on;
    plot(xlim, xlim, 'r--');
    xlabel('真实值'); ylabel('预测值');
    title(['输出 ' num2str(i)]);
    grid on;
end