clear,clc,close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 10-Stanford-LMO
%%% This script: Run the full multi-task modelling pipeline by sequentially
%%% executing dataset-specific model scripts (MLR, PLSR, GPR, SVR, DT, RF,
%%% XGBoost, Bayes, BNN, KNN, ELM, BPNN, DNN, CNN). Each sub-script performs
%%% training/evaluation and saves its own results.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run dataset/model scripts in a clean workspace each time to avoid
% cross-script variable leakage and ensure reproducibility

clear,clc,close all
run('Dataset_1_MutiTask_MLR.m')

clear,clc,close all
run('Dataset_2_MutiTask_PLSR.m')

clear,clc,close all
run('Dataset_3_MutiTask_GPR.m')

clear,clc,close all
run('Dataset_4_MutiTask_SVR.m')

clear,clc,close all
run('Dataset_5_MutiTask_DT.m')

clear,clc,close all
run('Dataset_6_MutiTask_RF.m')

clear,clc,close all
run('Dataset_7_MutiTask_XGBoost.m')

clear,clc,close all
run('Dataset_8_MutiTask_Bayes.m')

clear,clc,close all
run('Dataset_9_MutiTask_BNN.m')

clear,clc,close all
run('Dataset_10_MutiTask_KNN.m')

clear,clc,close all
run('Dataset_11_MutiTask_ELM.m')

clear,clc,close all
run('Dataset_12_MutiTask_BPNN.m')

clear,clc,close all
run('Dataset_13_MutiTask_DNN.m')

clear,clc,close all
run('Dataset_14_MutiTask_CNN.m')