clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Conventional Full-Charge Relaxation Voltage vs Pulse-Inspection
%%%          Relaxation Voltage
%%% This script: Parse step indices in SCU3 Dataset #2 to estimate the
%%% relative timing offsets of pulse/relaxation segments, compute the mean
%%% normalized timing profile, and visualize it as a bar chart.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SCU3 Dataset #2
% Load structured single-cycle dataset
load("../OneCycle_2.mat")

tic
CountData = 0;
for IndexData = 1:length(OneCycle)
    if OneCycle(IndexData).Steps(end) == 30
        CountData = CountData+1;

        % For each step group, locate the first sample index of the target step
        % and compute a relative time/index offset based on a fixed sampling factor.
        for IndexStep = 1:14
            TempTime = find(OneCycle(IndexData).Steps == (IndexStep*2+1));

            % The last step uses a different offset rule (data-specific alignment)
            if IndexStep == 14
                MyTime(IndexStep,CountData) = TempTime(1)-30*13*9-120*9;
            else
                MyTime(IndexStep,CountData) = TempTime(1)-30*IndexStep*9;
            end
        end
    end
end

% Mean timing profile normalized by the final-step mean (printed to console)
PMyTine = mean(MyTime,2)/mean(MyTime(14,:),2)

% Visualization
figure,bar(PMyTine)
toc