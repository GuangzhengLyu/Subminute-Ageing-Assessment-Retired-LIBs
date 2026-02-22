# 8-Open-Source Datasets
## External Validation on 10 Public Battery Ageing Datasets

This folder performs external validation of the proposed subminute cross-dimensional ageing assessment framework using 10 open-source lithium-ion battery datasets covering mainstream cathode chemistries (NCA, NCM, LFP, LMO). For each dataset, raw files are converted into a unified `OneCycle_*` format, internal-state features are constructed, ageing assessment is performed under either PLSR or 14 data-driven estimators, and sorting consistency is evaluated using K-means-based grouping.

This module corresponds to Supplementary Fig. 23 to Supplementary Fig. 32.

---

## Folder Structure (Root)

```
1-Tongji-NCA
2-RWTH-NCA
3-SDU-NCM
4-SDU-NCM(new)
5-Tongji-NCM
6-RWTH-NCM
7-MIT-A123
8-Stanford-A123
9-SCU1-A123
10-Stanford-LMO
```

Each subfolder corresponds to one public dataset and is self-contained.

---

## Subfolder Structure (Per Dataset)

Each dataset folder follows the same internal structure:

```
1-Data/
2-Multiple-Estimators/
G-K-means/

Feature_ALL_<DATASET>.mat
OneCycle_<DATASET>.mat
PLSR_Result_*

<DATASET>_4_featureView.m
<DATASET>_5_MutiTask_PLSR.m
<DATASET>_6_resultView.m
<DATASET>_7_K_Means.m
```

Example (RWTH-NCA):

```
Feature_ALL_RWTH_NCA.mat
OneCycle_NCA_RWTH.mat
PLSR_Result_1_80_Y_Test_..._RWTH_...

RWTH_NCA_4_featureView.m
RWTH_NCA_5_MutiTask_PLSR.m
RWTH_NCA_6_resultView.m
RWTH_NCA_7_K_Means.m
```

---

## Inputs

### Raw data

- Downloaded from the corresponding public dataset sources  
- Stored and parsed inside `1-Data/`

### Required processed inputs (generated locally)

```
OneCycle_<DATASET>.mat
```

Unified structure containing one full cycle time-series and ageing trajectory summaries.

### Required feature file

```
Feature_ALL_<DATASET>.mat
```

Six internal-state features extracted from relaxation-voltage modelling:

```
Uoc, R0, R1, C1, R2, C2
```

---

## Workflow (Per Dataset Folder)

### Stage 1: Data Preprocessing (Raw → OneCycle)

Folder:

```
1-Data/
```

- Reads raw dataset files  
- Unifies measurement channels, units, and sampling steps  
- Extracts one-cycle time series and long-term ageing summary  

Output:

```
OneCycle_<DATASET>.mat
```

---

### Stage 2: Feature Construction

Script:

```
<DATASET>_4_featureView.m
```

- Extracts pulse-inspection relaxation / equivalent relaxation segments (dataset-dependent)  
- Fits RC model  
- Builds internal-state feature matrix  
- Saves `Feature_ALL_<DATASET>.mat`

Input:

```
OneCycle_<DATASET>.mat
```

Output:

```
Feature_ALL_<DATASET>.mat
```

---

### Stage 3: Multi-Task Ageing Assessment (PLSR)

Script:

```
<DATASET>_5_MutiTask_PLSR.m
```

- Performs six-task ageing assessment using PLSR  
- Uses identical normalization and evaluation protocol as in the main dataset  
- Saves prediction outputs as `PLSR_Result_*`

Input:

```
Feature_ALL_<DATASET>.mat
OneCycle_<DATASET>.mat
```

Output:

```
PLSR_Result_*
```

---

### Stage 4: Result Visualization

Script:

```
<DATASET>_6_resultView.m
```

- Loads `PLSR_Result_*`  
- Generates prediction vs ground truth plots  
- Summarizes accuracy statistics for six tasks  

Input:

```
PLSR_Result_*
```

Output:

- Figures for Supplementary Fig. 23 to Supplementary Fig. 32 (dataset-dependent)

---

### Stage 5: Sorting Accuracy Evaluation (K-means)

Script:

```
<DATASET>_7_K_Means.m
```

- Uses ageing assessment outputs to regroup cells  
- Applies group-weighted K-means clustering  
- Computes within-group trajectory dispersion as sorting consistency metric  
- Produces normalized sorting performance for different task combinations  

Input:

```
PLSR_Result_*
OneCycle_<DATASET>.mat
```

Output:

- Sorting consistency plots / statistics  

---

## Optional: 14-Estimator Benchmarking

Folder:

```
2-Multiple-Estimators/
```

- Implements the same 14 estimators used in `7-Multiple-Estimators`  
- Runs ageing assessment with each estimator on this dataset  
- Produces estimator-wise accuracy and (if enabled) efficiency comparisons  

This submodule is used when the external validation requires comparing the proposed PLSR pipeline against other estimators on public datasets.

---

## Recommended Execution Order (Per Dataset)

1. `1-Data/` → generate `OneCycle_<DATASET>.mat`  
2. `<DATASET>_4_featureView.m` → generate `Feature_ALL_<DATASET>.mat`  
3. `<DATASET>_5_MutiTask_PLSR.m` → generate `PLSR_Result_*`  
4. `<DATASET>_6_resultView.m` → generate evaluation plots  
5. `<DATASET>_7_K_Means.m` → generate sorting consistency results  
6. (Optional) `2-Multiple-Estimators/` → benchmark 14 estimators  

---

## Comparison Principle

Across all 10 datasets, the validation is performed under a consistent pipeline:

- Unified data format (`OneCycle_*`)  
- Same internal-state feature definition (`Uoc, R0, R1, C1, R2, C2`)  
- Same six ageing tasks  
- Same normalization and evaluation metrics  
- Same sorting consistency metric (within-group trajectory dispersion)  

Only the dataset source and chemistry differ.

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
