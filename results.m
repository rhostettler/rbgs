% Rao-Blackwellized Gaussian filtering and smoothing example
% 
% Visualization of the simulation results as in [1].
%
% [1] R. Hostettler and S. Sarkka, "Rao–Blackwellized Gaussian smoothing," 
%     IEEE Transactions on Automatic Control, 2019
%
% 2016-present -- Roland Hostettler

clear variables;
load('results/results.mat');

%% Filtering results
figure(1); clf();
subplot(211);
plot(dzs, rmse_filter);
grid on;
ylim([0, 2]);
legend('UKF', 'RB-UKF', 'GHF', 'RB-GHF', 'RB-PF', 'Location', 'northeastoutside');
title('Filter RMSE');
xlabel('dz');
ylabel('e_{RMS}');

subplot(212);
semilogy(dzs, dt_filter);
legend('UKF', 'RB-UKF', 'GHF', 'RB-GHF', 'RB-PF', 'Location', 'northeastoutside');
grid on;
ylim([1e-4, 1e1]);
title('Filter run time');
xlabel('dz');
ylabel('t / s');

%% Smoothing results
figure(2); clf();
subplot(211);
plot(dzs, rmse_smoother);
grid on;
ylim([0, 2]);
legend('URTSS', 'RB-URTSS', 'GHS', 'RB-GHS', 'RB-FFBSi', 'Location', 'northeastoutside');
title('Smoother RMSE');
xlabel('dz');
ylabel('e_{RMS}');

subplot(212);
semilogy(dzs, dt_smoother);
grid on;
ylim([1e-4, 1e3]);
legend('URTSS', 'RB-URTSS', 'GHS', 'RB-GHS', 'RB-FFBSi', 'Location', 'northeastoutside');
title('Smoother run time');
xlabel('dz');
ylabel('t / s');
