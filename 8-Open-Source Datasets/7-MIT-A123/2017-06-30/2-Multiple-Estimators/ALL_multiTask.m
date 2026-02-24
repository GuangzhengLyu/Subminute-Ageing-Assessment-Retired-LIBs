clear,clc,close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123
%%% This script: Execute the model-specific multi-task pipelines in sequence
%%% (MLR, PLSR, GPR, SVR, DT, RF, XGBoost, Bayes, BNN, KNN, ELM, BPNN, DNN, CNN),
%%% clearing the workspace between runs to avoid cross-script contamination.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run 1: Multi-task MLR
run('Dataset_1_MutiTask_MLR.m')

clear,clc,close all
% Run 2: Multi-task PLSR
run('Dataset_2_MutiTask_PLSR.m')

clear,clc,close all
% Run 3: Multi-task GPR
run('Dataset_3_MutiTask_GPR.m')

clear,clc,close all
% Run 4: Multi-task SVR
run('Dataset_4_MutiTask_SVR.m')

clear,clc,close all
% Run 5: Multi-task Decision Tree (DT)
run('Dataset_5_MutiTask_DT.m')

clear,clc,close all
% Run 6: Multi-task Random Forest (RF)
run('Dataset_6_MutiTask_RF.m')

clear,clc,close all
% Run 7: Multi-task XGBoost
run('Dataset_7_MutiTask_XGBoost.m')

clear,clc,close all
% Run 8: Multi-task Bayesian regression/classifier baseline
run('Dataset_8_MutiTask_Bayes.m')

clear,clc,close all
% Run 9: Multi-task Bayesian Neural Network (BNN)
run('Dataset_9_MutiTask_BNN.m')

clear,clc,close all
% Run 10: Multi-task KNN
run('Dataset_10_MutiTask_KNN.m')

clear,clc,close all
% Run 11: Multi-task Extreme Learning Machine (ELM)
run('Dataset_11_MutiTask_ELM.m')

clear,clc,close all
% Run 12: Multi-task Backpropagation Neural Network (BPNN)
run('Dataset_12_MutiTask_BPNN.m')

clear,clc,close all
% Run 13: Multi-task Deep Neural Network (DNN)
run('Dataset_13_MutiTask_DNN.m')

clear,clc,close all
% Run 14: Multi-task Convolutional Neural Network (CNN)
run('Dataset_14_MutiTask_CNN.m')