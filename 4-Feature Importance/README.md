# 4-Feature Importance  
## SHAP-Based Feature Importance Analysis

This folder performs feature importance analysis using SHAP (Shapley values) to quantify how the six internal-state features contribute to the prediction of selected ageing targets. The implementation follows the MathWorks shapley workflow: create an explainer using a background set, compute local SHAP for a query point, and compute global importance as mean absolute SHAP over multiple query points.

This module corresponds to Supplementary Fig. 12.

---

## Folder Structure

SCU_SHAP_1.m  
SCU_SHAP_2.m  
SCU_SHAP_3.m  
SHAP_Example.m  

---

## Inputs

Each SCU_SHAP_k.m script loads the processed dataset and the feature file from the parent directory:

- ../OneCycle_1.mat and ../Feature_1_ALL.mat
- ../OneCycle_2.mat and ../Feature_2_ALL.mat
- ../OneCycle_3.mat and ../Feature_3_ALL.mat

The feature tensors in Feature_*_ALL.mat are expected to contain the six internal-state features:

- Uoc
- R0
- R1
- C1
- R2
- C2

The scripts use the 13th sampling terminal voltage (index 13) as the feature snapshot:

X = squeeze(Feature(:,13,:));

This corresponds to the high-voltage sampling point used for analysis consistency.

---

## Targets and Normalization

For each dataset, the script reconstructs per-sample labels from OneCycle and converts them into normalized health indicators:

- Capacity-based SOH: Capa/3.5
- RUL proxy: cycle count to a capacity threshold (dataset-dependent)
- Expanded indicators:
  - energy efficiency: ERate/89
  - CCC rate: CoChRate/83
  - mid-point voltage: (MindVolt-2.65)/(3.47-2.65)
  - platform discharge capacity: PlatfCapa/1.3

Then each target is further linearly scaled to [0,1] using dataset-specific Min_Out and Max_Out, and clipped into [0,1].

---

## SHAP Prediction Target Used in Each Script

- SCU_SHAP_1.m (Dataset #1):  
  y = Output(:,6) → platform-discharge-capacity-based SOH (normalized)

- SCU_SHAP_2.m (Dataset #2):  
  y = Output(:,6) → platform-discharge-capacity-based SOH (normalized)

- SCU_SHAP_3.m (Dataset #3):  
  y = Output(:,5) → mid-point-voltage-based SOH (normalized)

This design allows the feature-attribution analysis to focus on different service-relevant indicators under different retirement scenarios.

---

## Model and Explainer

A bagged-tree regression model is used as the SHAP carrier model:

mdl = fitrensemble(..., "Method","Bag", "NumLearningCycles",200, "Learners","Tree");

SHAP is computed using the Statistics and Machine Learning Toolbox:

- Background set: up to 200 random samples from training data
- Local explanation: one selected test sample
- Global importance: mean absolute SHAP over up to 100 random test samples

---

## Outputs

Each SCU_SHAP_k.m script produces:

1. Quick predictive sanity check  
   - prints test RMSE on a hold-out split (20%)

2. Local SHAP plot (one test sample)  
   - bar chart of SHAP contributions per feature (plot(explainer1))

3. Global SHAP importance  
   - table MeanAbsoluteShapley (mean(|SHAP|))  
   - bar chart of global importance

4. SHAP summary distribution  
   - swarmchart(explainerAll) for per-feature SHAP distributions

5. Dependence plot  
   - plotDependence(explainerAll, featureName) for feature-value vs SHAP

No files are saved by default unless you add explicit save/exportgraphics commands.

---

## Recommended Usage

Run dataset-specific scripts independently:

1. SCU_SHAP_1.m
2. SCU_SHAP_2.m
3. SCU_SHAP_3.m

If you want to reproduce Supplementary Fig. 12 consistently, use the same:

- random seed (rng)
- background-set size
- number of query points for global SHAP

---

## SHAP_Example.m

SHAP_Example.m is a standalone demonstration of the same workflow on synthetic data. It is used for validation and debugging of the SHAP pipeline and is not required for reproducing the manuscript results.

---

## Requirements

- MATLAB R2025B
- Statistics and Machine Learning Toolbox (fitrensemble, shapley, swarmchart, plotDependence)

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
