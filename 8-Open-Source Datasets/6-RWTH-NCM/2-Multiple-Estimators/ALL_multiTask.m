clear,clc,close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 6-RWTH-NCM
%%% This script: Run a multi-model benchmarking pipeline by sequentially
%%% calling dataset-specific multi-task learning scripts. Each run starts
%%% from a clean workspace (clear/clc/close all) to avoid cross-script state
%%% contamination and to improve reproducibility of the overall workflow.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dataset 1: Multi-task MLR
run('Dataset_1_MutiTask_MLR.m')

clear,clc,close all
% Dataset 2: Multi-task PLSR
run('Dataset_2_MutiTask_PLSR.m')

clear,clc,close all
% Dataset 3: Multi-task GPR
run('Dataset_3_MutiTask_GPR.m')

clear,clc,close all
% Dataset 4: Multi-task SVR
run('Dataset_4_MutiTask_SVR.m')

clear,clc,close all
% Dataset 5: Multi-task DT
run('Dataset_5_MutiTask_DT.m')

clear,clc,close all
% Dataset 6: Multi-task RF
run('Dataset_6_MutiTask_RF.m')

clear,clc,close all
% Dataset 7: Multi-task XGBoost
run('Dataset_7_MutiTask_XGBoost.m')

clear,clc,close all
% Dataset 8: Multi-task Bayes
run('Dataset_8_MutiTask_Bayes.m')

clear,clc,close all
% Dataset 9: Multi-task BNN
run('Dataset_9_MutiTask_BNN.m')

clear,clc,close all
% Dataset 10: Multi-task KNN
run('Dataset_10_MutiTask_KNN.m')

clear,clc,close all
% Dataset 11: Multi-task ELM
run('Dataset_11_MutiTask_ELM.m')

clear,clc,close all
% Dataset 12: Multi-task BPNN
run('Dataset_12_MutiTask_BPNN.m')

clear,clc,close all
% Dataset 13: Multi-task DNN
run('Dataset_13_MutiTask_DNN.m')

clear,clc,close all
% Dataset 14: Multi-task CNN
run('Dataset_14_MutiTask_CNN.m')