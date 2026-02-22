# 5-Number of Components
## PLSR Latent Component Sensitivity Study

This folder evaluates how the number of latent components in Partial Least Squares Regression (PLSR) affects ageing assessment accuracy. The same features, targets, and cross-validation protocol are used across all experiments, while only the PLSR component number is changed to isolate its impact.

This module corresponds to Supplementary Fig. 13.

---

## Folder Structure (Root)

```
1/
2/
3/
4/
5/
6/

SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

- Subfolders `1` to `6` correspond to PLSR component numbers `ncomp = 1` to `6`.
- The three `SCU3_6_resultView_*.m` scripts summarize and visualize results for Dataset #1 to #3.

---

## Subfolder Structure (Example)

Each component folder `{c}/` contains:

```
SCU3_5_MutiTask_PLSR{c}_1.m
SCU3_5_MutiTask_PLSR{c}_2.m
SCU3_5_MutiTask_PLSR{c}_3.m

SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

Notes:

- `{c}` is the fixed component number used inside:
  ```
  plsregress(..., ncomp)
  ```
- Suffix `_1/_2/_3` correspond to Dataset #1 / #2 / #3.

---

## Inputs

All scripts require processed datasets and features located in the parent directory:

```
../../OneCycle_1.mat
../../Feature_1_ALL.mat

../../OneCycle_2.mat
../../Feature_2_ALL.mat

../../OneCycle_3.mat
../../Feature_3_ALL.mat
```

The features are constructed from six internal-state parameters:

```
Uoc, R0, R1, C1, R2, C2
```

The scripts evaluate 13 sampling terminal voltages (3.0–4.2 V, step 0.1 V) by iterating:

```
CountSV = 13 : -1 : 1
```

---

## Ageing Tasks (Outputs)

Each PLSR model jointly predicts six targets (normalized then inverse-transformed):

1. Capacity-based SOH  
2. RUL (cycle count to a threshold)  
3. Energy-efficiency-based SOH  
4. CCC-rate-based SOH  
5. Mid-point-voltage-based SOH  
6. Platform-discharge-capacity-based SOH  

---

## Functional Modules

### Stage 1: Multi-Task PLSR with Fixed Component Number

```
SCU3_5_MutiTask_PLSR{c}_1.m
SCU3_5_MutiTask_PLSR{c}_2.m
SCU3_5_MutiTask_PLSR{c}_3.m
```

What it does:

- Normalizes features and outputs
- For each sampling voltage index (13 total):
  - Performs leave-one-out cross-validation (LOOCV)
  - Repeats prediction 100 times
- Saves prediction tensors `Y_Test` for each sampling voltage into files:

```
PLSR{c}_Result_..._Y_Test_%d.mat
```

(Exact filename pattern follows the script’s `sprintf` definition.)

Output (per sampling voltage):

```
Y_Test  % size ≈ [100, 6, N]
```

(100 repeats × 6 tasks × N samples)

---

### Stage 2: Accuracy Summary and Visualization

```
SCU3_6_resultView_1.m
SCU3_6_resultView_2.m
SCU3_6_resultView_3.m
```

What it does:

- Loads stored `Y_Test` files for all 13 sampling voltages
- Computes RMSE for each task and each repeat
- Aggregates RMSE across repeats and tasks
- Produces summary curves for component-number comparison

---

## Recommended Execution Order

For each component folder `{c} = 1 to 6`:

1. Run ageing assessment:
   ```
   SCU3_5_MutiTask_PLSR{c}_1.m
   SCU3_5_MutiTask_PLSR{c}_2.m
   SCU3_5_MutiTask_PLSR{c}_3.m
   ```

2. Summarize results:
   ```
   SCU3_6_resultView_1.m
   SCU3_6_resultView_2.m
   SCU3_6_resultView_3.m
   ```

3. Compare summary metrics across `{c} = 1 to 6` to reproduce Supplementary Fig. 13.

---

## Comparison Principle

To ensure fair comparison across component numbers:

- Same datasets (`OneCycle_1` to `3`)
- Same feature set (six internal-state parameters)
- Same sampling-voltage sweep (13 points)
- Same LOOCV protocol
- Same number of repeats (100)
- Only the PLSR latent component number differs

---

## Notes on Runtime and Storage

- The pipeline is computationally intensive because it nests:

```
13 sampling voltages × 100 repeats × N leave-one-out fits
```

- Each sampling voltage saves one `.mat` file containing `Y_Test`.
- Re-running scripts will overwrite existing `PLSR{c}_Result_*` files.

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
