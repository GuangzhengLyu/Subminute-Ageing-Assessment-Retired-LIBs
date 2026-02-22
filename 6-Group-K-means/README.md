# 6-Group-K-means
## Sorting Accuracy Evaluation via Group-Weighted K-means

This folder evaluates the sorting accuracy of retired batteries based on ageing assessment outputs. Batteries are grouped using a group-weighted K-means algorithm, and the sorting quality is quantified by the within-group dispersion of ageing trajectories (lower dispersion indicates higher sorting accuracy).

The analysis compares sorting performance under different ageing-assessment task combinations (capacity-only vs adding RUL, internal-state features, and expanded-indicator SOHs). This module corresponds to **Supplementary Fig. 14 to Supplementary Fig. 16**.

---

## Folder Structure

```
Group_K_Means/

SCU_K_Means_1.m
SCU_K_Means_2.m
SCU_K_Means_3.m
```

Group_K_Means/ provides:

- group-wise z-score normalization
- grouped weighted distance
- K-means++ initialization + Lloyd iterations (kmeans_pp_lloyd)

---

## Inputs

Each script loads:

- Processed ageing dataset: ../OneCycle_x.mat
- Internal-state features: ../Feature_x_ALL.mat
- Multi-task ageing assessment predictions from 1-Proposed:
  - Dataset #1: ../1-Proposed/PLSR_Result_1_70_Y_Test_13.mat
  - Dataset #2: ../1-Proposed/PLSR_Result_2_60_Y_Test_13.mat
  - Dataset #3: ../1-Proposed/PLSR_Result_3_50_Y_Test_13.mat

---

## What Is Clustered

The clustering feature vector is built from four feature groups (Xg{1}-Xg{4}), then group-wise normalized and concatenated.

### Group definition (as implemented)

- Group 1 (always included)  
  `Xg{1} = Estimation(:,1)`  
  Capacity-based SOH estimate (N×1)

- Group 2 (optional)  
  `Xg{2} = Estimation(:,2)`  
  RUL estimate (N×1)

- Group 3 (optional)  
  `Xg{3} = squeeze(Feature(:,13,:))`  
  Six internal-state features at 4.2 V (N×6)

- Group 4 (optional)  
  `Xg{4} = [Estimation(:,3:6)]`  
  Four expanded-indicator SOHs (N×4)

### Four sorting settings (Label = 1…4)

The scripts loop over `Kind = 1:4` and progressively add groups:

1. Label 1: SOH only (Group 1)  
2. Label 2: SOH + RUL (Group 1-2)  
3. Label 3: SOH + RUL + internal features (Group 1-3)  
4. Label 4: SOH + RUL + internal features + expanded SOHs (Group 1-4)

For groups not included, the script explicitly fills them with zeros to keep the same dimensional structure during concatenation.

---

## Group-Weighted K-means Configuration

- Group weights: w = [1, 1, 1, 1]
- Initialization: K-means++
- Optimizer: Lloyd iterations
- Repeats: opts.nInit = 8
- Max iterations: opts.maxIter = 200
- Convergence tolerance: opts.tol = 1e-6
- Random seed fixed per Kind: rng(42) (reproducible)

The clustering is executed in the full concatenated feature space; PCA is only used for visualization.

---

## Sorting Accuracy Metric

After clustering, each cluster is evaluated by computing the standard deviation across samples at each ageing-cycle index for multiple trajectories stored in `OneCycle(i).Cycle`:

- DiscCapaAh (discharge capacity trajectory)
- DiscEnergyWh (discharge energy trajectory)
- CharTimeS (charge time trajectory; filtered by isfinite)
- PlatfCapaAh (platform discharge capacity trajectory)

### Computation (as implemented)

For each cluster:

1. Align trajectories by using the shortest available length within that cluster.
2. For each cycle index m, compute std across cells in the cluster.
3. Average std(m) across m to get one dispersion value per cluster.
4. Average across all clusters to get one dispersion score per Label.

Finally, each dispersion score is normalized by the Label-1 baseline:

`Result(i,:) = M_STD_*Sum ./ M_STD_*Sum(1);`

So:

- Result(:,1) = 1 (baseline sorting using SOH only)
- Values < 1 indicate improved sorting accuracy relative to capacity-only sorting.

The scripts visualize Result using bar plots for each trajectory metric.

---

## Outputs

Running SCU_K_Means_x.m produces:

- K-means diagnostics in the console (SSE, iterations, cluster counts, mean intra-cluster distance)
- PCA scatter plot of cluster assignments
- Cluster-center “profile” plots per feature group
- Bar plots of normalized dispersion scores (Result) for:
  - discharge capacity
  - discharge energy
  - charge time
  - platform discharge capacity

---

## Recommended Usage

Run the scripts independently for each dataset:

1. SCU_K_Means_1.m
2. SCU_K_Means_2.m
3. SCU_K_Means_3.m

Ensure the 1-Proposed PLSR result files exist in the specified paths before running.

---

## Interpretation

This module evaluates whether richer ageing assessment outputs improve engineering sorting:

- Capacity-only sorting can group cells with similar SOH but different internal degradation paths.
- Adding RUL, internal-state features, and expanded-indicator SOHs is expected to reduce within-group trajectory dispersion, indicating improved sorting accuracy for cascade utilization.

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
