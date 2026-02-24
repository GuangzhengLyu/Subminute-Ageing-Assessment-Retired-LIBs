%% SHAP (Shapley) Example - MATLAB R2025B
% Fixes:
%   - Use S.Intercept instead of S.BaseValue (BaseValue does not exist)
%   - Use S.Shapley (table) + S.MeanAbsoluteShapley (table) as official outputs
%
% Requirements: Statistics and Machine Learning Toolbox

clear; clc; close all;
rng(7);

%% ========== 1) Prepare data (replace with your own) ==========
n = 800;   % samples
p = 6;     % features

X = randn(n,p);
y = 2*X(:,1) - 1.5*(X(:,2).^2) + 0.8*sin(X(:,3)) + 0.2*randn(n,1);

featureNames = "f" + string(1:p);

% Convert to table (recommended for shapley: keeps predictor names)
Tbl = array2table(X, "VariableNames", cellstr(featureNames));
Tbl.y = y;

% Train-test split
cv = cvpartition(n,"Holdout",0.2);
TblTr = Tbl(training(cv),:);
TblTe = Tbl(test(cv),:);

predictorNames = cellstr(featureNames);

Xtr = TblTr(:, predictorNames);
ytr = TblTr.y;
Xte = TblTe(:, predictorNames);
yte = TblTe.y;

%% ========== 2) Train a regression model ==========
mdl = fitrensemble(Xtr, ytr, ...
    "Method","Bag", ...
    "NumLearningCycles",200, ...
    "Learners","Tree");

% Quick check
yhat = predict(mdl, Xte);
rmse = sqrt(mean((yte - yhat).^2));
fprintf("Test RMSE = %.4f\n", rmse);

%% ========== 3) Create shapley object (NO SHAP computed yet) ==========
% Background set: use part of training data to speed up
bgN = min(200, height(Xtr));
bgIdx = randperm(height(Xtr), bgN);
Xbg = Xtr(bgIdx,:);

explainer = shapley(mdl, Xbg);  % does NOT compute SHAP until you call fit()
% (This matches MathWorks workflow: create object -> fit for query points) :contentReference[oaicite:3]{index=3}

%% ========== 4) Local explanation for ONE query point ==========
sampleID = 1;                 % pick one test sample
xq1 = Xte(sampleID,:);        % 1×p table row

explainer1 = fit(explainer, xq1);    % compute SHAP for this query point :contentReference[oaicite:4]{index=4}

% Local SHAP table: Predictor | Value
localShapTbl = explainer1.Shapley;   % table with per-feature contributions :contentReference[oaicite:5]{index=5}

% Base/average prediction (baseline)
baseValue = explainer1.Intercept;    % baseline (average prediction) :contentReference[oaicite:6]{index=6}

% Model prediction for this point
predValue = predict(mdl, xq1);

% Reconstruct prediction: base + sum(SHAP)
reconValue = baseValue + sum(localShapTbl.Value);

fprintf("\n--- Local SHAP (one sample) ---\n");
fprintf("Intercept (baseline)       = %.6f\n", baseValue);
fprintf("predict(mdl, x)            = %.6f\n", predValue);
fprintf("Intercept + sum(SHAP)      = %.6f\n", reconValue);

% Plot local SHAP bar chart (official)
figure("Name","Local SHAP (one query point)");
plot(explainer1);  % bar graph of SHAP values for this query point :contentReference[oaicite:7]{index=7}

%% ========== 5) Global importance across MULTIPLE query points ==========
% Choose multiple query points (subset of test set)
qN = min(100, height(Xte));
qIdx = randperm(height(Xte), qN);
Xq = Xte(qIdx,:);

explainerAll = fit(explainer, Xq);   % compute SHAP for many query points :contentReference[oaicite:8]{index=8}

% Global importance: mean(|SHAP|) across query points
globalTbl = explainerAll.MeanAbsoluteShapley; % Predictor | Value :contentReference[oaicite:9]{index=9}

fprintf("\n--- Global SHAP importance (mean(|SHAP|)) Top 10 ---\n");
disp(globalTbl(1:min(10,height(globalTbl)),:));

% Plot global importance (official: bar graph of mean abs SHAP)
figure("Name","Global SHAP importance");
plot(explainerAll);  % for multiple query points: plots mean absolute SHAP :contentReference[oaicite:10]{index=10}

% Optional: summary distribution (swarm chart)
figure("Name","SHAP summary (swarmchart)");
swarmchart(explainerAll);     % distribution across query points :contentReference[oaicite:11]{index=11}

% Optional: dependence plot for one feature
featureToInspect = predictorNames{1};
figure("Name","SHAP dependence");
plotDependence(explainerAll, featureToInspect);  % feature value vs SHAP :contentReference[oaicite:12]{index=12}

%% ========== 6) How to plug in your engineering data ==========
% Replace the synthetic part with:
%   Tbl = array2table(yourX, "VariableNames", yourFeatureNames);
%   Tbl.y = yourY;
% Keep the rest unchanged.
