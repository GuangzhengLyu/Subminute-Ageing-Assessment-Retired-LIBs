clear,clc,close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: External Validation on 10 Open-Source Battery Ageing Datasets
%%% Dataset: 7-MIT-A123
%%% This script: Sequentially run the multi-task modelling pipelines across
%%% multiple algorithms (MLR, PLSR, GPR, SVR, DT, RF, XGBoost, Bayes, BNN,
%%% KNN, ELM, BPNN, DNN, CNN) with a clean workspace between runs.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run multi-task pipelines (clear workspace before each run for isolation)
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