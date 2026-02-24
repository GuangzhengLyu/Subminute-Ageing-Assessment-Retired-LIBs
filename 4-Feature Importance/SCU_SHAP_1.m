clear;
clc;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: SHAP-Based Feature Importance Analysis
%%% This script: Build SCU3 Dataset #1 samples and normalized health outputs,
%%% then train a bagged-tree regression model and compute local/global SHAP
%%% (Shapley) feature contributions using MathWorks shapley workflow.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SCU3 Dataset #1
% Load structured single-cycle dataset and pre-extracted relaxation features
load('../OneCycle_1.mat')
load('../Feature_1_ALL.mat')

%% Feature tensor assembly
% Feature(:,:,k) stores the k-th relaxation-derived parameter across
% samples (row) and voltage setpoints (column).
Feature(:,:,1) = Uoc;
Feature(:,:,2) = R0;
Feature(:,:,3) = R1;
Feature(:,:,4) = C1;
Feature(:,:,5) = R2;
Feature(:,:,6) = C2;

%% Sample construction
% Filter samples by the ending step flag and extract life, original capacity,
% and expanded health indicators (used as multi-dimensional outputs).
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % Define life as the first cycle where discharge capacity drops below 2.5 Ah
        % If the threshold is not reached, use the full trajectory length
        if (min(OneCycle(IndexData).Cycle.DiscCapaAh) <= 2.5)
            Life(CountData,1) = min(find(OneCycle(IndexData).Cycle.DiscCapaAh < 2.5));
        else
            Life(CountData,1) = length(OneCycle(IndexData).Cycle.DiscCapaAh);
        end

        % Original capacity (Ah)
        Capa(CountData,1) = OneCycle(IndexData).OrigCapaAh;

        % Expanded health indicators (cycle-index selection follows raw data structure)
        ERate(CountData,1) = OneCycle(IndexData).Cycle.EnergyRate(2);
        CoChRate(CountData,1) = OneCycle(IndexData).Cycle.ConstCharRate(2);
        MindVolt(CountData,1) = OneCycle(IndexData).Cycle.MindVoltV(1);
        PlatfCapa(CountData,1) = OneCycle(IndexData).Cycle.PlatfCapaAh(1);
    end
end

%% Unified health-indicator scaling
% Convert raw variables into comparable normalized health-indicator forms
% (these are not the final [0,1] min-max normalization used for learning).
Capa = Capa/3.5;
Life = Life;
ERate = ERate/89;
CoChRate = CoChRate/83;
MindVolt = (MindVolt-2.65)/(3.47-2.65);
PlatfCapa = PlatfCapa/1.3;

%% Output normalization
% Apply min-max normalization to construct the multi-dimensional output matrix
Max_Out = [0.99, 450, 1.01, 1.02, 1.04, 1.05];
Min_Out = [0.79, 100, 0.94, 0.9,  0.86, 0.65];

Output(:,1) = (Capa-Min_Out(1))/(Max_Out(1)-Min_Out(1));
Output(:,2) = (Life-Min_Out(2))/(Max_Out(2)-Min_Out(2));
% Expanded health indicators
Output(:,3) = (ERate-Min_Out(3))/(Max_Out(3)-Min_Out(3));
Output(:,4) = (CoChRate-Min_Out(4))/(Max_Out(4)-Min_Out(4));
Output(:,5) = (MindVolt-Min_Out(5))/(Max_Out(5)-Min_Out(5));
Output(:,6) = (PlatfCapa-Min_Out(6))/(Max_Out(6)-Min_Out(6));

% Clip outputs to [0,1]
Output(Output<0) = 0;Output(Output>1) = 1;

%% Data preparation for SHAP analysis
% Select one voltage setpoint (here: setpoint index = 13) and one target output.
% X:  N×6 feature matrix (Uoc, R0, R1, C1, R2, C2)
% y:  N×1 target vector (here: Output(:,6) = platform discharge capacity indicator)
X = squeeze(Feature(:,13,:));
y = Output(:,6);

% Predictor names used by shapley (table variable names are recommended)
featureNames = ["Uoc","R0","R1","C1","R2","C2"];

% Convert to table (recommended for shapley: preserves predictor names)
Tbl = array2table(X, "VariableNames", cellstr(featureNames));
Tbl.y = y;

%% Train-test split
% Hold out 20% of samples for a quick sanity-check evaluation
cv = cvpartition(size(X,1),"Holdout",0.2);
TblTr = Tbl(training(cv),:);
TblTe = Tbl(test(cv),:);

predictorNames = cellstr(featureNames);

Xtr = TblTr(:, predictorNames);
ytr = TblTr.y;
Xte = TblTe(:, predictorNames);
yte = TblTe.y;

%% Train a regression model
% Bagged decision trees (Random Forest-style ensemble)
mdl = fitrensemble(Xtr, ytr, ...
    "Method","Bag", ...
    "NumLearningCycles",200, ...
    "Learners","Tree");

%% Quick evaluation on the held-out set
yhat = predict(mdl, Xte);
rmse = sqrt(mean((yte - yhat).^2));
fprintf("Test RMSE = %.4f\n", rmse);

%% Create shapley object
% Background set: use a subset of training data to reduce SHAP cost.
% Note: shapley() does not compute SHAP until fit() is called.
bgN = min(200, height(Xtr));
bgIdx = randperm(height(Xtr), bgN);
Xbg = Xtr(bgIdx,:);

explainer = shapley(mdl, Xbg);

%% Local explanation for one query point
% Compute SHAP values for a single test sample
sampleID = 1;
xq1 = Xte(sampleID,:);

explainer1 = fit(explainer, xq1);

% Local SHAP table: Predictor | Value (per-feature contribution)
localShapTbl = explainer1.Shapley;

% Baseline prediction (intercept / expected value over background)
baseValue = explainer1.Intercept;

% Model prediction for this point
predValue = predict(mdl, xq1);

% Reconstruction check: baseline + sum(SHAP)
reconValue = baseValue + sum(localShapTbl.Value);

fprintf("\n--- Local SHAP (one sample) ---\n");
fprintf("Intercept (baseline)       = %.6f\n", baseValue);
fprintf("predict(mdl, x)            = %.6f\n", predValue);
fprintf("Intercept + sum(SHAP)      = %.6f\n", reconValue);

% Plot local SHAP bar chart (one query point)
figure("Name","Local SHAP (one query point)");
plot(explainer1);

%% Global importance across multiple query points
% Compute SHAP values for multiple test samples and summarize importance
qN = min(100, height(Xte));
qIdx = randperm(height(Xte), qN);
Xq = Xte(qIdx,:);

explainerAll = fit(explainer, Xq);

% Global importance: mean(|SHAP|) across query points
globalTbl = explainerAll.MeanAbsoluteShapley;

fprintf("\n--- Global SHAP importance (mean(|SHAP|)) Top 10 ---\n");
disp(globalTbl(1:min(10,height(globalTbl)),:));

% Plot global importance (mean absolute SHAP)
figure("Name","Global SHAP importance");
plot(explainerAll);

%% Optional visual diagnostics
% Summary distribution (swarm chart) across query points
figure("Name","SHAP summary (swarmchart)");
swarmchart(explainerAll);

% Dependence plot: feature value vs SHAP contribution for one predictor
featureToInspect = predictorNames{1};
figure("Name","SHAP dependence");
plotDependence(explainerAll, featureToInspect);

%% Notes for adapting to other datasets / targets
% Replace:
%   X = squeeze(Feature(:,setpointIndex,:));
%   y = Output(:,targetIndex);
% Keep the remaining workflow unchanged (table conversion, model training,
% shapley background selection, and local/global explanations).