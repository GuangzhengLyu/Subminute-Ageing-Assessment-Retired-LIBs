# 7-Multiple-Estimators
## Benchmarking 14 Data-Driven Estimators for Multi-Task Ageing Assessment

This folder benchmarks 14 data-driven estimators under an identical ageing-assessment setting. Each estimator is used to solve the same six ageing tasks using the same six internal-state features extracted from relaxation-voltage modeling. The goal is to compare assessment accuracy and assessment time across estimators under a controlled protocol.

This module corresponds to Supplementary Fig. 17 to Supplementary Fig. 22 and Supplementary Table 1 to Supplementary Table 6.

---

## Folder Structure (Root)

```
1-GPR
1-MLR
1-PLSR
1-SVR
2-BayesRegression
2-BNN
2-DT
2-ELM
2-KNN
2-RF
2-XGBoost
3-BPNN
3-CNN
3-DNN
```

- Prefix 1/2/3 denotes the estimator family category used in the study.
- Each subfolder is self-contained and follows the same internal workflow.

---

## Subfolder Structure

Each estimator subfolder contains:

```
SCU3_5_MutiTask_<EST>_1.m
SCU3_5_MutiTask_<EST>_2.m
SCU3_5_MutiTask_<EST>_3.m

SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

Notes:

- `<EST>` is the estimator identifier (e.g., GPR, SVR, RF, DNN).
- Suffix `_1/_2/_3` correspond to Dataset #1 / #2 / #3.

---

## Inputs

All estimator scripts require the same processed datasets and features (located in the parent directory):

- `../../OneCycle_1.mat`, `../../Feature_1_ALL.mat`
- `../../OneCycle_2.mat`, `../../Feature_2_ALL.mat`
- `../../OneCycle_3.mat`, `../../Feature_3_ALL.mat`

The scripts construct the six internal-state features:

```
Uoc, R0, R1, C1, R2, C2
```

and evaluate them at multiple sampling terminal voltages (13 points from 3.0 V to 4.2 V), by iterating:

```
CountSV = 1 to 13
```

---

## Ageing Tasks (Outputs)

Each estimator predicts the same six targets:

1. Capacity-based SOH  
2. RUL (cycle-life to a dataset-specific threshold)  
3. Energy-efficiency-based SOH  
4. CCC-rate-based SOH  
5. Mid-point-voltage-based SOH  
6. Platform-discharge-capacity-based SOH  

Targets are normalized to [0,1] during training and inverse-transformed back to engineering units for evaluation.

---

## Workflow

### Stage 1: Multi-Task Estimation with a Fixed Estimator

```
SCU3_5_MutiTask_<EST>_1.m
SCU3_5_MutiTask_<EST>_2.m
SCU3_5_MutiTask_<EST>_3.m
```

- Normalizes features and outputs  
- For each sampling voltage index (13 total):
  - performs leave-one-out cross-validation (LOOCV)
  - repeats training/testing multiple times (e.g., 100 repeats) for stability  
- Saves prediction results as `Y_Test` for each sampling voltage:

```
<EST>_Result_*_Y_Test_<CountSV>.mat
```

Example (GPR, Dataset #1):

```
GPR_Result_1_70_Y_Test_13.mat
```

Output:

- `Y_Test`: repeated predictions for 6 tasks across all samples  

---

### Stage 2: Result Summary and Visualization

```
SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

- Loads the saved `Y_Test` results for all sampling voltages  
- Computes prediction error statistics (e.g., RMSE) for each task  
- Summarizes the estimator’s overall performance for that dataset  

---

## Recommended Execution Order

For each estimator subfolder:

1. Run the estimator for Dataset #1 to #3  
   ```
   SCU3_5_MutiTask_<EST>_1.m
   SCU3_5_MutiTask_<EST>_2.m
   SCU3_5_MutiTask_<EST>_3.m
   ```

2. Summarize results  
   ```
   SCU3_6_resultView_1.m
   SCU3_6_resultView_2.m
   SCU3_6_resultView_3.m
   ```

3. Compare summary metrics across all 14 estimators to reproduce Supplementary Fig. 17 to Supplementary Fig. 22 and Supplementary Table 1 to Supplementary Table 6.

---

## Comparison Principle

To ensure fair benchmarking:

- Same datasets (OneCycle_1 to 3)  
- Same six internal-state features  
- Same 13-point sampling-voltage sweep  
- Same LOOCV protocol and repeat count  
- Same six prediction tasks and the same normalization/inverse-normalization  
- Only the estimator type differs  

---

## Notes

- Some estimators (e.g., GPR, BNN, DNN/CNN) can be significantly slower than linear models under LOOCV.  
- Scripts may overwrite existing `*_Result_*` files if re-run.  
- If benchmarking evaluation time, keep timing logic consistent across estimators (e.g., `tic/toc` around model fitting + prediction) and record the same unit.

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
