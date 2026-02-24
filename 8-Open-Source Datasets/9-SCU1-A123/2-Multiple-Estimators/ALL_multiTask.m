clear,clc,close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 9-SCU1-A123
%%% This script: Run the full multi-task modelling benchmark by sequentially
%%% executing dataset-specific scripts for different regression models.
%%% Each sub-script is run in a clean MATLAB workspace for consistency.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1) Multi-task MLR
clear,clc,close all
run('Dataset_1_MutiTask_MLR.m')

% 2) Multi-task PLSR
clear,clc,close all
run('Dataset_2_MutiTask_PLSR.m')

% 3) Multi-task GPR
clear,clc,close all
run('Dataset_3_MutiTask_GPR.m')

% 4) Multi-task SVR
clear,clc,close all
run('Dataset_3_MutiTask_SVR.m')

% 5) Multi-task Decision Tree
clear,clc,close all
run('Dataset_5_MutiTask_DT.m')

% 6) Multi-task Random Forest
clear,clc,close all
run('Dataset_6_MutiTask_RF.m')

% 7) Multi-task XGBoost
clear,clc,close all
run('Dataset_7_MutiTask_XGBoost.m')

% 8) Multi-task Bayesian Regression
clear,clc,close all
run('Dataset_8_MutiTask_Bayes.m')

% 9) Multi-task BNN
clear,clc,close all
run('Dataset_9_MutiTask_BNN.m')

% 10) Multi-task KNN
clear,clc,close all
run('Dataset_10_MutiTask_KNN.m')

% 11) Multi-task ELM
clear,clc,close all
run('Dataset_11_MutiTask_ELM.m')

% 12) Multi-task BPNN
clear,clc,close all
run('Dataset_12_MutiTask_BPNN.m')

% 13) Multi-task DNN
clear,clc,close all
run('Dataset_13_MutiTask_DNN.m')

% 14) Multi-task CNN
clear,clc,close all
run('Dataset_14_MutiTask_CNN.m')