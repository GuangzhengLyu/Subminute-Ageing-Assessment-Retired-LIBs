# 0-Data Process
## Construction of Unified OneCycle Datasets from Raw Experimental Data

This folder converts raw experimental ageing data into the unified `OneCycle_*` format used throughout this repository.

Three subfolders correspond to the three consecutive retirement scenarios:

```
1-Dataset#1
2-Dataset#2
3-Dataset#3
```

Each dataset follows a slightly different raw-data structure; therefore, preprocessing scripts are dataset-specific.

---

## Important Note on Raw Data Storage

Due to storage limitations, each dataset folder contains only one battery sample as an example.

The complete raw experimental data are available from Zenodo:

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

To construct the complete dataset:

1. Download all raw data parts from Zenodo.  
2. Replace the example raw files in `1-Raw Data`.  
3. Repeat the preprocessing procedure for each battery sample.  

---

## Folder Structure (Per Dataset)

Example: `1-Dataset#1`

```
1-Raw Data
2-Cycle
3-Record
4-Battery

DataProcess_1_Battery_1.m
DataProcess_2_OneCycle_1.m
```

Dataset #2 and #3 follow the same structure, with dataset-specific script names:

```
DataProcess_1_Battery_2.m
DataProcess_2_OneCycle_2.m
DataProcess_1_Sample_3.m
DataProcess_2_OneCycle_3.m
```

---

## Data Processing Workflow

The construction pipeline consists of two stages.

---

### Stage 1: Raw Data → Battery Structure

Script:

```
DataProcess_1_Battery_*.m
```

Purpose:

- Organize time-series signals (time, current, voltage, etc.)  
- Group data by battery and by cycle  
- Generate structured `.mat` battery files  

Input:

- Intermediate files inside `2-Cycle`  
- Intermediate files inside `3-Record`  

Output:

```
4-Battery/
```

The battery structure stores:

- Full cycle time-series data  
- Charge/discharge summaries  
- Per-cycle performance indicators  

---

### Stage 2: Battery → OneCycle Dataset

Script:

```
DataProcess_2_OneCycle_*.m
```

Purpose:

- Extract a single complete charge–discharge cycle for relaxation analysis  
- Extract long-term ageing trajectory indicators  
- Construct unified structure:

```
OneCycle(i).OrigCapaAh
OneCycle(i).CurrentA
OneCycle(i).VoltageV
OneCycle(i).Cycle.*
```

Output:

```
OneCycle_1.mat
OneCycle_2.mat
OneCycle_3.mat
```

These files are the direct inputs for all subsequent modules:

```
1-Proposed
2-RV and PIRV
3-RC Models
5-Number of Components
6-Group-K-means
7-Multiple-Estimators
```

---

## Dataset-Specific Notes

### Dataset #1

- Standard structure  
- Includes Raw Data, Cycle, Record, and Battery layers  

Processing scripts:

```
DataProcess_1_Battery_1.m
DataProcess_2_OneCycle_1.m
```

---

### Dataset #2

- Similar structure to Dataset #1  
- Minor differences in record organization and indexing  

Processing scripts:

```
DataProcess_1_Battery_2.m
DataProcess_2_OneCycle_2.m 
```

---

### Dataset #3

- Different raw-data structure  
- Uses `Sample` instead of `Record`  

Processing scripts:

```
DataProcess_1_Sample_3.m
DataProcess_2_OneCycle_3.m
```

Additional preprocessing adjustments are implemented to handle:

- Data alignment  
- Missing fields  
- Sample-based organization  

---

## Recommended Execution Order (Per Dataset)

1. Place raw data into `1-Raw Data`  
2. Run `DataProcess_1_*`  
3. Verify generated files in `4-Battery`  
4. Run `DataProcess_2_OneCycle_*`  
5. Confirm `OneCycle_*.mat` is generated  

---

## Reproducibility

All downstream experiments depend exclusively on:

```
OneCycle_1.mat
OneCycle_2.mat
OneCycle_3.mat
```

Once these files are constructed, raw data are no longer required for:

- Feature extraction  
- Ageing assessment  
- Sorting consistency evaluation  
- Estimator benchmarking  

---

## Citation

If you use the dataset or code in this repository, please cite the associated work:

Lyu, G., Tao, S., Zhang, H., Goetz, S. M., Zio, E. & Miao, Q.  
Subminute diagnostics reveal hidden heterogeneity of deep ageing patterns beyond capacity for second-life lithium-ion batteries.
