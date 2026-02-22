# 3-RC Models  
## RC Model Order Study for Feature Extraction and Ageing Assessment

This folder implements the comparative study of RC equivalent-circuit model orders (from 1RC to 5RC) for retired-battery subminute diagnostics.

For each RC order, the pipeline is kept identical except for the model order used to fit relaxation voltage. Therefore, differences in:

- Feature extraction efficiency (runtime)
- Model fitting accuracy (RMSE of relaxation-voltage fitting)
- Downstream ageing assessment accuracy (multi-task prediction performance)

can be attributed to RC order only.

This module corresponds to **Supplementary Fig. 10–11** (RC order comparison).

---

## Folder Structure

1RC/

2RC/

3RC/

4RC/

5RC/


Each subfolder is self-contained and follows the same internal structure.

---

## Subfolder Structure

Each `{k}RC/` folder contains:

### (A) Extracted Feature Files

Feature_1_ALL_{k}RC.mat

Feature_2_ALL_{k}RC.mat

Feature_3_ALL_{k}RC.mat


### (B) Scripts

SCU3_4_featureView_1_{k}RC.m

SCU3_4_featureView_2_{k}RC.m

SCU3_4_featureView_3_{k}RC.m

SCU3_5_MutiTask_PLSR_1_{k}RC.m

SCU3_5_MutiTask_PLSR_2_{k}RC.m

SCU3_5_MutiTask_PLSR_3_{k}RC.m

SCU3_6_resultView_1.m

SCU3_6_resultView_2.m

SCU3_6_resultView_3.m


**Notes:**

- `{k} = 1, 2, 3, 4, 5` indicates RC order.
- `SCU3_6_resultView_*.m` is shared for visualization and reused across RC orders.

---

## Inputs and Outputs

### Required Inputs (Shared Across All RC Orders)

- `OneCycle_1.mat`
- `OneCycle_2.mat`
- `OneCycle_3.mat`

(located in the parent directory)

---

### Feature Outputs (Per RC Order)

`Feature_x_ALL_{k}RC.mat` contains RC-identified internal-state parameters at **13 sampling terminal voltages (3.0–4.2 V)**.

- For **1RC**, the features are typically:
  - `Uoc`
  - `R0`
  - `R1`
  - `C1`

- For **kRC**, the features expand to:
  - `Uoc`
  - `R0`
  - `{R_i, C_i}` for `i = 1 ... k`

---

### Ageing Assessment Outputs (Per RC Order)

Multi-task PLSR scripts save prediction results as:

PLSR_Result_*_{k}RC.mat

(Depending on implementation, these files may be saved in the current folder or an internal `Result/` directory. Maintain a consistent structure for visualization.)

---

## Functional Modules (Per `{k}RC` Folder)

Suffix `_1`, `_2`, `_3` correspond to Dataset #1, #2, and #3.

---

### Stage 1: RC-Based Feature Extraction + Model Fitting Accuracy  
`SCU3_4_featureView_1/2/3_{k}RC.m`

- Extracts relaxation voltage segments at multiple sampling voltages
- Fits the `{k}RC` model to relaxation-voltage decay
- Outputs:
  - RC parameters used as features
  - Fitting RMSE (for model-order accuracy comparison)
  - Runtime measured by `tic/toc` (for efficiency comparison)

**Input:**

- `OneCycle_x.mat`

**Output:**

- `Feature_x_ALL_{k}RC.mat`
- Console statistics for mean fitting RMSE and timing

---

### Stage 2: Multi-Task Ageing Assessment (PLSR)  
`SCU3_5_MutiTask_PLSR_1/2/3_{k}RC.m`

- Uses `{k}RC` features
- Performs the same multi-task ageing assessment protocol for fair comparison

**Tasks (consistent across RC orders):**

1. Capacity-based SOH  
2. RUL  
3. Energy-efficiency-based SOH  
4. CCC-rate-based SOH  
5. Mid-point-voltage-based SOH  
6. Platform-capacity-based SOH  

- Uses leave-one-out cross-validation
- Repeats prediction (e.g., 100 runs) for statistical stability

**Input:**

- `Feature_x_ALL_{k}RC.mat`
- `OneCycle_x.mat` (for labels and indicator extraction)

**Output:**

- `PLSR_Result_*_{k}RC.mat`

---

### Stage 3: Result Visualization (Assessment Accuracy Comparison)  
`SCU3_6_resultView_1/2/3.m`

- Loads PLSR prediction outputs
- Visualizes prediction vs ground truth and error statistics
- Used to compare:
  - Accuracy across RC orders
  - Dispersion and robustness across datasets

**Input:**

- `PLSR_Result_*_{k}RC.mat`

---

## Recommended Execution Order (Per RC Order)

For each `{k}RC/` folder and each dataset:

1. `SCU3_4_featureView_{x}_{k}RC`
2. `SCU3_5_MutiTask_PLSR_{x}_{k}RC`
3. `SCU3_6_resultView_{x}`

Repeat for `{k} = 1 ... 5`, then aggregate results to reproduce the RC-order comparison in **Supplementary Fig. 10–11**.

---

## Comparison Principle

To isolate the effect of RC order, all non-model factors are held constant:

- Same datasets (`OneCycle_1–3`)
- Same sampling terminal voltages (13 points from 3.0 to 4.2 V)
- Same downstream estimator (multi-task PLSR)
- Same cross-validation protocol
- Same evaluation metrics

Only the RC model order differs.

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
*Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.*
