# Subminute-Ageing-Assessment-Retired-LIBs

🚨 **IMPORTANT — THIS REPOSITORY IS NOT SELF-CONTAINED**

This GitHub repository does **NOT** include the three processed ageing datasets required to execute the code:

- `OneCycle_1.mat`
- `OneCycle_2.mat`
- `OneCycle_3.mat`

To reproduce the reported results, please:

1. Download the required datasets from Zenodo  👉 https://zenodo.org/records/18694513
2. Place them in the root directory: **Subminute-Ageing-Assessment-Retired-LIBs/**

Alternatively, you may directly download the **full ready-to-run package** from the same Zenodo record.

---

## Manuscript Information

This repository accompanies the manuscript:

**“Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries”**  

and its Supplementary Information.

This work addresses rapid and cross-dimensional ageing assessment of retired lithium-ion batteries under consecutive retirements and deep ageing conditions. A three-year ageing dataset is reported, covering:

- 590 samples  
- 119,691 cycles  
- Over 3.25 billion time-series entries  

across unexpected, normal, and secondary retirement scenarios.

Beyond capacity, four service-relevant performance indicators are analysed:

- Energy efficiency  
- Constant-current charge rate  
- Mid-point voltage  
- Platform discharge capacity  

A subminute assessment framework is developed using pulse-inspection relaxation voltage and a second-order RC equivalent circuit model. Six internal-state features extracted from 30 seconds of relaxation voltage data enable joint assessment of:

- SOH  
- RUL  
- Expanded SOH indicators  

via partial least squares regression.

All code and processed data required to reproduce the reported results are provided in this repository.

---

## 1. Project Overview

This repository implements the full experimental and computational pipeline for:

- Deep-ageing dataset construction under three consecutive retirement scenarios  
- Expanded performance indicator analysis  
- Pulse-inspection relaxation voltage (PIRV)  
- Second-order RC equivalent circuit parameter extraction  
- Subminute cross-dimensional ageing assessment  
- Multi-model comparison (14 estimators)  
- Sorting-accuracy validation  
- Cross-chemistry external validation on 10 public datasets  

All experiments are implemented in: **MATLAB R2025B**

---

## 2. Repository Structure

```
0-Data Process
1-Proposed
2-RV and PIRV
3-RC Models
4-Feature Importance
5-Number of Components
6-Group-K-means
7-Multiple-Estimators
8-Open-Source Datasets

OneCycle_1.mat
OneCycle_2.mat
OneCycle_3.mat

Feature_1_ALL.mat
Feature_2_ALL.mat
Feature_3_ALL.mat
```

---

## 3. Folder-Level Description (Figure Mapping Included)

---

### 0-Data Process

Generates three processed datasets:

```
OneCycle_1.mat  % Unexpected retirement
OneCycle_2.mat  % Normal retirement
OneCycle_3.mat  % Secondary retirement
```

Each `OneCycle_x.mat` contains:

- Full voltage-current-time sequence of a complete charge-discharge cycle  
- Ageing trajectory (cycle index)  
- Expanded performance indicators  

These datasets serve as the base input for all subsequent modules.

---

### 1-Proposed (Main Framework)

Implements the core method described in the manuscript.

Includes:

- Deep-ageing observation  
- Expanded indicator construction  
- Pulse-inspection relaxation voltage extraction  
- Second-order RC modeling  
- Internal-state feature extraction  
- Six-task multi-output PLSR assessment  
- Result visualization  

Reproduces: **Main Text Fig. 1 to Fig. 5**

---

### 2-RV and PIRV

Comparison between:

- Conventional full-charge relaxation voltage (RV)  
- Pulse-inspection relaxation voltage (PIRV)  

Evaluates:

- Measurement time reduction  
- Assessment accuracy  

Reproduces: **Supplementary Fig. 8 and Supplementary Fig. 9**

---

### 3-RC Models

Comparison of RC model order:

```
1RC to 5RC
```

For each order:

- Feature extraction  
- Modeling RMSE  
- Computational time  
- Downstream ageing assessment accuracy  

Reproduces: **Supplementary Fig. 10 and Supplementary Fig. 11**

---

### 4-Feature Importance

SHAP-based feature contribution analysis.

Evaluates:

- Contribution of six internal-state parameters  
- Task-wise feature relevance  

Reproduces: **Supplementary Fig. 12**

---

### 5-Number of Components

PLSR latent component sensitivity study. **Components tested: 1 to 6**

Reproduces: **Supplementary Fig. 13**

---

### 6-Group-K-means

Engineering sorting validation workflow:

1. Load ageing assessment results  
2. K-means clustering  
3. Performance trajectory dispersion evaluation  
4. Sorting accuracy metric  

Reproduces: **Supplementary Fig. 14 to Supplementary Fig. 16**

---

### 7-Multiple-Estimators

Implements 14 regression-based estimators.

Regression & Statistical:

```
MLR
GPR
SVR
PLSR
```

Conventional Machine Learning:

```
DT
RF
XGBoost
Bayesian Regression
BNN
KNN
ELM
```

Deep Learning:

```
BPNN
DNN
CNN
```

Each model solves six tasks:

1. Capacity-based SOH  
2. RUL  
3. Energy-efficiency-based SOH  
4. CCC-rate-based SOH  
5. Mid-point-voltage-based SOH  
6. Platform-capacity-based SOH  

Reproduces: **Supplementary Fig. 17 to Supplementary Fig. 22** and **Supplementary Table 1 to Supplementary Table 6**

---

### 8-Open-Source Datasets

Validation on 10 open-source datasets.

Chemistries:

```
NCA
NCM
LFP
LMO
```

For each dataset:

- Data preprocessing  
- Relaxation voltage and internal state feature extraction  
- Ageing assessment  
- Sorting validation  

Reproduces: **Supplementary Fig. 23 to Supplementary Fig. 32**

---

## 4. Shared MAT Files

### OneCycle_1/2/3.mat

Contain:

- Full-cycle time series  
- Ageing trajectory  
- Expanded indicators  

Used by:

```
1-Proposed
2-RV and PIRV
3-RC Models
4-Feature Importance
5-Number of Components
6-Group-K-means
7-Multiple-Estimators
```

---

### Feature_1/2/3_ALL.mat

Contain:

- Extracted internal-state features  

Used by:

```
1-Proposed
3-RC Models
4-Feature Importance
5-Number of Components
6-Group-K-means
7-Multiple-Estimators
```

---

## 5. Figure-to-Code Mapping

```
1-Proposed                Fig. 1-5                
2-RV and PIRV             Sup. Fig. 8-9           
3-RC Models               Sup. Fig. 10-11         
4-Feature Importance      Sup. Fig. 12            
5-Number of Components    Sup. Fig. 13            
6-Group-K-means           Sup. Fig. 14-16         
7-Multiple-Estimators     Sup. Fig. 17-22     Sup. Table 1-6      
8-Open-Source Datasets    Sup. Fig. 23-32         
```

---

## 6. Data Availability

Three-year ageing datasets of retired lithium-ion batteries under consecutive retirement scenarios are reported, covering:

- 590 samples  
- 119,691 distinct cycles  
- 3.25 billion unique data entries  
- 440 GB raw experimental data  

Raw data and code deposited in Zenodo:

```
Dataset Part 1:
https://zenodo.org/records/18633811
Dataset Part 2:
https://zenodo.org/records/17904751
Dataset Part 3:
https://zenodo.org/records/17912803
Dataset Part 4:
https://zenodo.org/records/17912818
Dataset Part 5:
https://zenodo.org/records/17912831
Dataset Part 6:
https://zenodo.org/records/17913097
Dataset Part 7:
https://zenodo.org/records/17913616
Dataset Part 8:
https://zenodo.org/records/17913646
```

Processed datasets and feature files required to reproduce all figures are included in this repository.

---

## 7. Code Availability

All scripts required to reproduce:

- Internal-state modeling  
- Feature construction  
- Multi-task ageing assessment  
- Estimator comparison  
- Sorting validation  
- Cross-chemistry validation  

are provided in this repository.

---

## 8. Computational Environment

```
MATLAB R2025B
Intel Core i9-12900K
Leave-one-out cross-validation
Random seeds fixed where applicable
Large MAT files saved using -v7.3
```

---

## 9. Reproducibility Workflow (Minimal Path)

To reproduce main results:

1. Run `1-Proposed`  
2. Run `6-Group-K-means` for sorting validation  
3. Execute estimator comparison if needed (`7-Multiple-Estimators`)  

---

## 10. Citation

If you use the dataset or code in this repository, please cite:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
