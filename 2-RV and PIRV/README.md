# 2-RV and PIRV
## Conventional Full-Charge Relaxation Voltage vs Pulse-Inspection Relaxation Voltage

This folder implements the comparative study between:

- Conventional full-charge relaxation voltage (RV)
- Pulse-inspection relaxation voltage (PIRV, implemented in 1-Proposed)

The purpose of this module is to:

1. Extract relaxation voltage after complete CC-CV charging  
2. Construct equivalent internal-state features  
3. Perform identical multi-task ageing assessment  
4. Compare assessment accuracy and measurement efficiency  
5. Quantify time reduction achieved by PIRV  

This module reproduces the results shown in **Supplementary Fig. 8 and Supplementary Fig. 9**.

---

## Folder Structure

```
Result/

Feature_1_EC.mat
Feature_2_EC.mat
Feature_3_EC.mat

SCU3_4_featureView_1.m
SCU3_4_featureView_2.m
SCU3_4_featureView_3.m

SCU3_4_TimeMeasure_1.m
SCU3_4_TimeMeasure_2.m
SCU3_4_TimeMeasure_3.m

SCU3_5_MutiTask_PLSR_1.m
SCU3_5_MutiTask_PLSR_2.m
SCU3_5_MutiTask_PLSR_3.m

SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

---

## Methodological Difference from 1-Proposed

In contrast to PIRV (30 s relaxation inserted during charging):

- RV is measured only once  
- Measured after full CC-CV charging  
- Requires long CV charging duration  
- Cannot sample multiple voltage levels  
- Has lower excitation richness  

The downstream modelling and assessment pipeline remains identical to ensure fair comparison.

---

## Functional Modules

Suffix `_1`, `_2`, `_3` correspond to Dataset #1, #2, and #3.

---

## Stage 1: Feature Extraction Based on RV

```
SCU3_4_featureView_1.m
SCU3_4_featureView_2.m
SCU3_4_featureView_3.m
```

- Extracts relaxation voltage after full charge  
- Fits equivalent circuit model  
- Generates internal-state parameters  
- Saves features as:

```
Feature_1_EC.mat
Feature_2_EC.mat
Feature_3_EC.mat
```

Input:

```
OneCycle_x.mat
```

Output:

```
Feature_x_EC.mat
```

---

## Stage 2: Measurement Time Evaluation

```
SCU3_4_TimeMeasure_1.m
SCU3_4_TimeMeasure_2.m
SCU3_4_TimeMeasure_3.m
```

- Computes full measurement duration required for RV  
- Compares with PIRV measurement time  
- Quantifies time reduction ratio  
- Supports **Supplementary Fig. 9**  

---

## Stage 3: Multi-Task Ageing Assessment

```
SCU3_5_MutiTask_PLSR_1.m
SCU3_5_MutiTask_PLSR_2.m
SCU3_5_MutiTask_PLSR_3.m
```

- Uses RV-based features  
- Implements identical PLSR structure as 1-Proposed  
- Performs leave-one-out cross-validation  
- Saves predicted and true values  

Input:

```
Feature_x_EC.mat
```

Output:

```
Result/
```

---

## Stage 4: Comparative Visualization

```
SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

- Compares:
  - Prediction accuracy  
  - Error distributions  
  - Statistical performance  
- Directly contrasts FCRV and PIRV assessment results  

Input:

```
Results from Stage 3
PIRV results (from 1-Proposed)
```

Corresponds to: **Supplementary Fig. 8 and Supplementary Fig. 9**

---

## Result Folder

The `Result/` directory stores:

- RV-based assessment outputs  
- Predicted vs true values  

Existing files may be overwritten when rerunning scripts.

---

## Recommended Execution Order

For each dataset:

```
1. SCU3_4_featureView
2. SCU3_4_TimeMeasure
3. SCU3_5_MutiTask_PLSR
4. SCU3_6_resultView
```

---

## Key Comparison Principle

To ensure fair comparison:

- Same datasets  
- Same six ageing tasks  
- Same PLSR configuration  
- Same cross-validation protocol  
- Only relaxation acquisition strategy differs  

This isolates the impact of relaxation strategy on:

- Measurement efficiency  
- Feature quality  
- Ageing assessment accuracy  

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
