%% I. 清空环境变量
clear all
clc
%% II. 导入数据
% load concrete_data.mat

attributes =xlsread('力X.xlsx','A1:N315')';
strength=xlsread('力X.xlsx','O1:O315')';

%%
% 1. 随机产生训练集和测试集
w = randperm(size(attributes,2));
%  n=(1:315);
%%
% 2. 训练集――252个样本
p_train = attributes(:,w(1:252))';
t_train = strength(:,w(1:252))';
 
%%
% 3. 测试集――63个样本
p_test = attributes(:,w(253:end))';
t_test = strength(:,w(253:end))';
 
%% III. 数据归一化
%%
% 1. 训练集
[pn_train,inputps] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',inputps);
pn_test = pn_test';
 
%%
% 2. 测试集
[tn_train,outputps] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',outputps);
tn_test = tn_test';
 
%% IV. SVM模型创建/训练(RBF核函数)
%%
% 1. 寻找最佳c参数/g参数
[c,g] = meshgrid(-10:0.5:10,-10:0.5:10);
[m,n] = size(c);
cg = zeros(m,n);  
eps = 10^(-3);
v = 2;
bestc = 0;
bestg = 0;
error = Inf;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];%2 高斯函数 -v返回一个具体地值；-s 3代表SVR回归
        cg(i,j) = svmtrain(tn_train,pn_train,cmd);
        if cg(i,j) < error
            error = cg(i,j);
            bestc = 2^c(i,j);%取出c
            bestg = 2^g(i,j);%取出g
        end
        if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)%c优先考虑
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
    end
end
 
%%
% 2. 创建/训练SVM  
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];%把最好的c和g拼接起来，作为参数项
model = svmtrain(tn_train,pn_train,cmd);
 
%% V. SVM仿真预测
[Predict_1,error_1,decision_values1] = svmpredict(tn_train,pn_train,model);
[Predict_2,error_2,decision_values2] = svmpredict(tn_test,pn_test,model);
%%
% 1. 反归一化
predict_1 = mapminmax('reverse',Predict_1,outputps);
predict_2 = mapminmax('reverse',Predict_2,outputps);
 
%%
% 2. 结果对比
result_1 = [t_train predict_1];
result_2 = [t_test predict_2];
 
%% VI. SVR训练集和测试集绘图
%%  训练集绘图1
figure(1);
plot(1:length(t_train),t_train,'r-*',1:length(t_train),predict_1,'b:o')
grid on
legend('真实值','预测值')
xlabel('训练集样本编号')
ylabel('磨损量')
string_1 = {'SVR训练值和真实值的对比';
           ['mse = ' num2str(error_1(2)) ' R^2 = ' num2str(error_1(3))]};
title(string_1)

%%  测试集绘图2
figure(2);
subplot(2,1,1);  % 图2包含2行1列个子图形，首先绘制子图1
plot(1:length(t_test),t_test,'-*b',1:length(t_test),predict_2,':og')
% 用蓝色的*绘制测试数据的真实值；用绿色的o绘制测试数据的预测值
hold on;
legend('真实值','预测值');% 子图1的注释
% title('SVR预测磨损量结果','fontsize',12)  %子图1的标题
xlabel('测试集样本编号','fontsize',12); % x轴
ylabel('磨损量','fontsize',12); % y轴
string_2 = {'SVR测试集预测结果对比';
           ['mse = ' num2str(error_2(2)) ' R^2 = ' num2str(error_2(3))]};
title(string_2)

subplot(2,1,2);  % 绘制子图2
plot(predict_2 - t_test,'-*r');  % 输出测试数据的预测误差
title('SVR预测磨损量误差','fontsize',12)  %子图2的标题
ylabel('误差','fontsize',12);  % y轴
xlabel('测试集样本编号','fontsize',12);  % x轴
% ylim([0,0.01]);
ylim auto