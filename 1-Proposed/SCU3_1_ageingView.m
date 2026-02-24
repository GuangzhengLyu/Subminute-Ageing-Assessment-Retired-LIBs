clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Project: Subminute Cross-Dimensional Ageing Assessment
%%% This script: Build index-based discharge-capacity trajectories for three
%%% datasets, then visualize the mean trajectory and ±1/2/3σ bands.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dataset #1
% Load single-cycle structure (OneCycle) and extract discharge-capacity
% sequences within the target capacity window.
load('../OneCycle_1.mat')
Temp_1 = OneCycle;

% Iterate cells in reverse order and construct labeled capacity index curves
Count_1 = 0;
for i = 105:-1:60
    Count_1 = Count_1+1;

    % Keep only the segment after the last point where discharge capacity > 3.2 Ah
    [A B] = find(Temp_1(i).Cycle.DiscCapaAh > 3.2);
    Capa_1{Count_1,1} = Temp_1(i).Cycle.DiscCapaAh(max(A):end-1);

    % Discretization resolution (Ah) for capacity window scanning
    Resolu = 0.01;

    % For each target capacity level j, compute the mean index of samples
    % whose capacity falls within [j-Resolu, j+Resolu)
    CountLebec = 0;
    for j = 3.2:(-1*Resolu):2.5
        CountLebec = CountLebec+1;

        % Index set selection by thresholding around j
        C= find(Capa_1{Count_1,1} >= j-Resolu);
        D= find(Capa_1{Count_1,1} <  j+Resolu);
        E= intersect(C,D);

        % Mean index represents the "location" of capacity level j in the curve
        LebeCapa_1(Count_1,CountLebec) = mean(E);

        % Handle missing bins by forward-filling (or zero for the first bin)
        if isnan(LebeCapa_1(Count_1,CountLebec))
            if CountLebec == 1
                LebeCapa_1(Count_1,CountLebec) = 0;
            else
                LebeCapa_1(Count_1,CountLebec) = LebeCapa_1(Count_1,CountLebec-1);
            end
        end
    end

    % Quick per-sample visualization of the labeled index curve
    figure(1),hold on,plot(LebeCapa_1(Count_1,:))
    axis([0,70,0,1200])
end

% Aggregate statistics across samples: mean and standard deviation by column
data_1 = LebeCapa_1(:,1:70);
m_1 = mean(data_1, 1); s_1 = std(data_1, 0, 1); x_1 = 1:length(m_1);

% Plot mean curve with ±1σ/±2σ/±3σ uncertainty bands
figure(4); hold on;
fill([x_1, fliplr(x_1)], [m_1+3*s_1, fliplr(m_1-3*s_1)], [1,0.8,0.8], 'EdgeColor','none');
fill([x_1, fliplr(x_1)], [m_1+2*s_1, fliplr(m_1-2*s_1)], [1,1,0.8], 'EdgeColor','none');
fill([x_1, fliplr(x_1)], [m_1+s_1, fliplr(m_1-s_1)], [0.8,0.8,1], 'EdgeColor','none');
plot(m_1, 'k-', 'LineWidth', 2);
xlabel('Variable index'); ylabel('Value'); grid on; legend('3\sigma','2\sigma','1\sigma','Mean');
axis([0,70,0,2200])

%% Dataset #2
% Repeat the same processing for Dataset #2 with its own capacity window
load('../OneCycle_2.mat')
Temp_2 = OneCycle;

Count_2 = 0;
for i = 46:-1:1
    Count_2 = Count_2+1;

    % Use the full discharge-capacity sequence directly
    Capa_2{Count_2,1} = Temp_2(i).Cycle.DiscCapaAh;

    Resolu = 0.01;
    CountLebec = 0;
    for j = 2.5:(-1*Resolu):2.1
        CountLebec = CountLebec+1;

        C= find(Capa_2{Count_2,1} >= j-Resolu);
        D= find(Capa_2{Count_2,1} <  j+Resolu);
        E= intersect(C,D);
        LebeCapa_2(Count_2,CountLebec) = mean(E);

        % Missing-bin handling consistent with Dataset #1
        if isnan(LebeCapa_2(Count_2,CountLebec))
            if CountLebec == 1
                LebeCapa_2(Count_2,CountLebec) = 0;
            else
                LebeCapa_2(Count_2,CountLebec) = LebeCapa_2(Count_2,CountLebec-1);
            end
        end
    end

    figure(2),hold on,plot(LebeCapa_2(Count_2,:))
    axis([0,40,0,1200])
end

% Align Dataset #2 trajectories to Dataset #1 endpoint for concatenation/offset
data_2 = LebeCapa_2(:,1:40)+m_1(end);
m_2 = mean(data_2, 1); s_2 = std(data_2, 0, 1); x_2 = 1:length(m_2);

figure(5); hold on;
fill([x_2, fliplr(x_2)], [m_2+3*s_2, fliplr(m_2-3*s_2)], [1,0.8,0.8], 'EdgeColor','none');
fill([x_2, fliplr(x_2)], [m_2+2*s_2, fliplr(m_2-2*s_2)], [1,1,0.8], 'EdgeColor','none');
fill([x_2, fliplr(x_2)], [m_2+s_2, fliplr(m_2-s_2)], [0.8,0.8,1], 'EdgeColor','none');
plot(m_2, 'k-', 'LineWidth', 2);
xlabel('Variable index'); ylabel('Value'); grid on; legend('3\sigma','2\sigma','1\sigma','Mean');
axis([0,40,0,2200])

%% Dataset #3
% Dataset #3 includes multiple channels; process only the last record of each
% channel segment in the reverse traversal.
load('../OneCycle_3.mat')
Temp_3 = OneCycle;

Count_3 = 0;
for i = 432:-1:1
    % Start of a new channel block (reverse order) or the first element
    if i == 432 || (Temp_3(i).Channel ~= Temp_3(i+1).Channel)
        Count_3 = Count_3+1;

        % If capacity reaches above 2.1 Ah, keep the tail after last > 2.1 Ah
        if max(Temp_3(i).Cycle.DiscCapaAh) >= 2.1
            [A B] = find(Temp_3(i).Cycle.DiscCapaAh > 2.1);
            Capa_3{Count_3,1} = Temp_3(i).Cycle.DiscCapaAh(max(A):end-1);
        else
            % Otherwise, use the full sequence
            Capa_3{Count_3,1} = Temp_3(i).Cycle.DiscCapaAh;
        end

        Resolu = 0.01;
        CountLebec = 0;
        for j = 2.1:(-1*Resolu):1.75
            CountLebec = CountLebec+1;

            C= find(Capa_3{Count_3,1} >= j-Resolu);
            D= find(Capa_3{Count_3,1} <  j+Resolu);
            E= intersect(C,D);
            LebeCapa_3(Count_3,CountLebec) = mean(E);

            % Missing-bin handling consistent with Datasets #1–#2
            if isnan(LebeCapa_3(Count_3,CountLebec))
                if CountLebec == 1
                    LebeCapa_3(Count_3,CountLebec) = 0;
                else
                    LebeCapa_3(Count_3,CountLebec) = LebeCapa_3(Count_3,CountLebec-1);
                end
            end
        end

        figure(3),hold on,plot(LebeCapa_3(Count_3,:))
        axis([0,40,0,1200])
    end
end

% Align Dataset #3 trajectories to Dataset #2 endpoint for concatenation/offset
data_3 = LebeCapa_3(:,1:35)+m_2(end);
m_3 = mean(data_3, 1); s_3 = std(data_3, 0, 1); x_3 = 1:length(m_3);

figure(6); hold on;
fill([x_3, fliplr(x_3)], [m_3+3*s_3, fliplr(m_3-3*s_3)], [1,0.8,0.8], 'EdgeColor','none');
fill([x_3, fliplr(x_3)], [m_3+2*s_3, fliplr(m_3-2*s_3)], [1,1,0.8], 'EdgeColor','none');
fill([x_3, fliplr(x_3)], [m_3+s_3, fliplr(m_3-s_3)], [0.8,0.8,1], 'EdgeColor','none');
plot(m_3, 'k-', 'LineWidth', 2);
xlabel('Variable index'); ylabel('Value'); grid on; legend('3\sigma','2\sigma','1\sigma','Mean');
axis([0,35,0,2200])