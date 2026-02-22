# Subminute-Ageing-Assessment-Retired-LIBs

This repository accompanies the manuscript:

**“Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries”**  
and its Supplementary Information.

This work addresses rapid and cross-dimensional ageing assessment of retired lithium-ion batteries under consecutive retirements and deep ageing conditions. A three-year ageing dataset is reported, covering **590 samples**, **119,691 cycles**, and over **3.25 billion time-series entries** across unexpected, normal, and secondary retirement scenarios.

Beyond capacity, four service-relevant performance indicators—energy efficiency, constant-current charge rate, mid-point voltage, and platform discharge capacity—are analysed to characterize deep ageing behaviour. A subminute assessment framework is developed using pulse-inspection relaxation voltage (PIRV) and a second-order RC equivalent circuit model. Six internal-state features extracted from 30 seconds of relaxation voltage data enable joint assessment of SOH, RUL, and expanded SOH indicators via partial least squares regression.

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
- Sorting-consistency validation  
- Cross-chemistry external validation on 10 public datasets  

All experiments are implemented in **MATLAB R2025B**.

---

## 2. Repository Structure
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

---

## 3. Folder-Level Description (Figure Mapping Included)

### 0-Data Process

Generates three processed datasets:

- `OneCycle_1.mat` – Unexpected retirement  
- `OneCycle_2.mat` – Normal retirement  
- `OneCycle_3.mat` – Secondary retirement  

Each `OneCycle_x` file contains:

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

Reproduces:

- **Main Text Fig. 1–5**

---

### 2-RV and PIRV

Comparison between:

- Conventional full-charge relaxation voltage (RV)  
- Pulse-inspection relaxation voltage (PIRV)

Evaluates:

- Measurement time reduction  
- Assessment accuracy  

Reproduces:

- **Supplementary Fig. 8–9**

---

### 3-RC Models

Comparison of RC model order (1–5 RC modules).

For each order:

- Feature extraction  
- Modeling RMSE  
- Computational time  
- Downstream ageing assessment accuracy  

Reproduces:

- **Supplementary Fig. 10–11**

---

### 4-Feature Importance

SHAP-based feature contribution analysis.

Evaluates:

- Contribution of six internal-state parameters  
- Task-wise feature relevance  

Reproduces:

- **Supplementary Fig. 12**

---

### 5-Number of Components

PLSR latent component sensitivity study.  
Components tested: 1–6  

Reproduces:

- **Supplementary Fig. 13**

---

### 6-Group-K-means

Engineering sorting validation workflow:

1. Loading ageing assessment results  
2. K-means clustering  
3. Performance trajectory dispersion evaluation  
4. Sorting consistency metric  

Reproduces:

- **Supplementary Fig. 14–16**

---

### 7-Multiple-Estimators

Implements 14 regression-based estimators.

**Regression & Statistical Models**

- MLR  
- GPR  
- SVR  
- PLSR  

**Conventional Machine Learning**

- DT  
- RF  
- XGBoost  
- Bayesian Regression  
- BNN  
- KNN  
- ELM  

**Deep Learning**

- BPNN  
- DNN  
- CNN  

Each model solves six tasks:

1. Capacity-based SOH  
2. RUL  
3. Energy-efficiency-based SOH  
4. CCC-rate-based SOH  
5. Mid-point-voltage-based SOH  
6. Platform-capacity-based SOH  

Reproduces:

- **Supplementary Fig. 17–22**  
- **Supplementary Table 1–6**

---

### 8-Open-Source Datasets

Validation on 10 open-source datasets.

Chemistries:

- NCA  
- NCM  
- LFP  
- LMO  

For each dataset:

- Data preprocessing  
- PIRV feature extraction  
- Ageing assessment  
- Sorting validation  

Reproduces:

- **Supplementary Fig. 23–32**

---

## 4. Shared MAT Files

### OneCycle_1/2/3.mat

Provide:

- Full-cycle time series  
- Ageing trajectory  
- Expanded indicators  

Loaded by scripts in:

- 1-Proposed  
- 2-RV and PIRV  
- 3-RC Models  
- 4-Feature Importance  
- 5-Number of Components  
- 6-Group-K-means  
- 7-Multiple-Estimators  

---

### Feature_1/2/3_ALL.mat

Provide:

- Extracted internal-state features  

Loaded by scripts in:

- 1-Proposed  
- 3-RC Models  
- 4-Feature Importance  
- 5-Number of Components  
- 6-Group-K-means  
- 7-Multiple-Estimators  

---

## 5. Figure-to-Code Mapping

| Figure | Code Folder |
|--------|------------|
| Fig. 1–5 | 1-Proposed |
| Sup. Fig. 8–9 | 2-RV and PIRV |
| Sup. Fig. 10–11 | 3-RC Models |
| Sup. Fig. 12 | 4-Feature Importance |
| Sup. Fig. 13 | 5-Number of Components |
| Sup. Fig. 14–16 | 6-Group-K-means |
| Sup. Fig. 17–22 | 7-Multiple-Estimators |
| Sup. Fig. 23–32 | 8-Open-Source Datasets |

---

## 6. Data Availability

The raw experimental data and code have been deposited in Zenodo.

The dataset includes:

- 590 samples  
- 119,691 cycles  
- 3.25 billion data entries  
- 440 GB raw experimental data  

Zenodo Records:

- Part 1: https://zenodo.org/records/18633811  
- Part 2: https://zenodo.org/records/17904751  
- Part 3: https://zenodo.org/records/17912803  
- Part 4: https://zenodo.org/records/17912818  
- Part 5: https://zenodo.org/records/17912831  
- Part 6: https://zenodo.org/records/17913097  
- Part 7: https://zenodo.org/records/17913616  
- Part 8: https://zenodo.org/records/17913646  

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

- MATLAB R2025B  
- Intel Core i9-12900K  
- Leave-one-out cross-validation  
- Random seeds fixed where applicable  
- Large MAT files saved using `-v7.3`

---

## 9. Minimal Reproducibility Path

To reproduce the main results:

1. Run `1-Proposed`
2. Run `6-Group-K-means` for sorting validation
3. Execute `7-Multiple-Estimators` if estimator benchmarking is required

---

## 10. Citation

If you use the dataset or code in this repository, please cite:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
*Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.*
