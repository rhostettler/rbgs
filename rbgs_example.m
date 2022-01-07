% Rao-Blackwellized Gaussian filtering and smoothing example
% 
% This is the example of Rao-Blackwellized Gaussian filtering/smoothing
% presented in [1].
%
% [1] R. Hostettler and S. Sarkka, "Rao–Blackwellized Gaussian smoothing," 
%     IEEE Transactions on Automatic Control, 2019
%
% 2016-present -- Roland Hostettler

%{
% Copyright (C) 2016-present Roland Hostettler
%
% This file is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
% 
% This file is distributed in the hope that it will be useful, but WITHOUT 
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
% FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
% more details.
% 
% You should have received a copy of the GNU General Public License along 
% with this file. If not, see <http://www.gnu.org/licenses/>.
%}

% Housekeeping
clear variables;
addpath lib;
rng(123);

%% Parameters
% Should the results be stored?
% N.B.: Use with caution, this overwrites any existing results
export = true;

% Methods to use
unscented = true;       % Unscented Kalman filter / smoother
gauss_hermite = true;   % Gauss-Hermite Kalman filter / smoother
particle = true;        % Particle filter / smoother (N.B.: This will take a long time)

% Unscented transform parameters
alpha = 1;
beta = 0;
kappa = 0;

% Gauss-Hermite quadrature parameters
p = 3;

% Particle filter/smooter parameters
Jf = 250;               % Filter
Js = 100;               % Smoother

% Simulation parameters
K = 100;                % No. of MC runs (100)
Ts = 0.05;              % Sampling time (0.05)
N = 100;                % No. of time samples (100)

% Model parameters
dz = 2;                 % No. of linear states
ds = 1;                 % No. of nonlinear states
dx = ds+dz;             % Total number of states
dy = 1;                 % Number of measurements
Q = 0.01*eye(dx);       % Process noise covariance
R = 0.1;                % Measurement noise variance

%% Model definition
% State indices
is = 1;
iz = 1+(1:dz).';

% Initial State
m0 = [
    2*pi*1;
    kron(ones(dz/2, 1), [1; 0])
];
P0 = eye(dx).^2;

% Generate the matrix CC which has cos(i*w*Ts) on the diagonal
i = 1:dz/2;
cc = zeros(1, dz);
cc(1:2:end) = i;
cc(2:2:end) = i;
CC = @(xn) diag(cos(cc*xn*Ts));

% Generate the matrix SS which has sin(-i*w*Ts) and sin(i*w*Ts) on the
% off-diagonals
ss = zeros(1, dz-1);
ss(1:2:end) = i;
SS = @(xn) diag(-sin(ss*xn*Ts), 1) + diag(sin(ss*xn*Ts), -1);

% Dynamic model (Rao-Blackwellized form)
% x[n] = f(s[n-1]) + A(s[n-1])*z[n-1] + q[n]
f = @(s) [
    s;
    zeros(dz, 1);
];
A = @(s) [
    zeros(ds, dz);
    CC(s) + SS(s);
];

% x[n] = ff(x[n-1]) + q[n]
ff = @(x) f(x(is)) + A(x(is))*x(iz);
Q = @(~) Q;

% Measurement Model
% y[n] = g(s[n]) + B(s[n])*z[n] + r[n]
g = @(~) zeros(dy, 1); 
B = @(~) kron(ones(1, dz/2), [1, 0]);

% y[n] = gg(x[n]) + r[n]
gg = @(x) g(x(is)) + B(x(is))*x(iz);
R = @(~) R;

%% Preallocate
% Simulated trajectories
xs = zeros(dx, N, K);
ys = zeros(dy, N, K);

% Unscented
t_ukf = zeros(1, K);
t_rbukf = zeros(1, K);
t_urtss = zeros(1, K);
t_rburtss = zeros(1, K);
m_ukf = zeros(dx, N, K);
m_rbukf = zeros(dx, N, K);
m_urtss = zeros(dx, N, K);
m_rburtss = zeros(dx, N, K);

% Gauss-Hermite
t_ghf = zeros(1, K);
t_rbghf = zeros(1, K);
t_ghs = zeros(1, K);
t_rbghs = zeros(1, K);
m_ghf = zeros(dx, N, K);
m_rbghf = zeros(dx, N, K);
m_ghs = zeros(dx, N, K);
m_rbghs = zeros(dx, N, K);

% Particle filter/smoother
t_rbpf = zeros(1, K);
t_rbffbsi = zeros(1, K);
m_rbpf = zeros(dx, N, K);
m_rbps = zeros(dx, N, K);

%% Simulation
% Monte Carlo runs
for k = 1:K
    str = sprintf('Monte Carlo run %d of %d...\n', k, K);
    fprintf(str);
    
    %% Generate data by simulating the system
    y = zeros(1, N);
    x = m0 + chol(P0)'*randn(dx, 1);

    for n = 1:N
        % Propagate the state
        q = chol(Q(x(is))).'*randn(dx, 1);
        x = ff(x) + q;

        % Measurement
        r = chol(R(x(is))).'*randn(dy, 1);
        y(:, n) = gg(x) + r;

        % Store
        xs(:, n, k) = x;
        ys(:, n, k) = y(:, n);
    end
    
    %% Unscented filter/smoother
    if unscented
        % UKF & URTSS
        [Xi, wm, wc] = ut_sigmapoints(zeros(dx, 1), eye(dx), alpha, beta, kappa);
        
        tic();
        [m, P, mp, Pp, Cp] = gf(ys(:, :, k), m0, P0, ff, Q, gg, R, Xi, wm, wc);
        t_ukf(:, k) = toc();

        tic();
        ms = rts_smooth(m, P, mp, Pp, Cp);
        t_urtss(:, k) = t_ukf(:, k) + toc();
        m_ukf(:, :, k) = m;
        m_urtss(:, :, k) = ms;
           
        % RB-UKF & RB-URTSS
        [Xi, wm, wc] = ut_sigmapoints(zeros(ds, 1), eye(ds), alpha, beta, kappa);
        tic;
        [m, P, mp, Pp, Cp] = rbgf(ys(:, :, k), m0, P0, f, A, Q, g, B, R, is, iz, Xi, wm, wc);
        t_rbukf(k) = toc();
        
        tic();
        ms = rts_smooth(m, P, mp, Pp, Cp);
        t_rburtss(k) = t_rbukf(k) + toc();
        m_rbukf(:, :, k) = m;
        m_rburtss(:, :, k) = ms;
    end
        
    %% Gauss-Hermite filter/smoother
    if gauss_hermite
        % GHF and GHS
        [Xi, w] = gh_sigmapoints(zeros(dx, 1), eye(dx), 3);
        
        tic();
        [m, P, mp, Pp, Cp] = gf(ys(:, :, k), m0, P0, ff, Q, gg, R, Xi, w);
        t_ghf(:, k) = toc();
        
        tic();
        ms = rts_smooth(m, P, mp, Pp, Cp);
        t_ghs(:, k) = t_ghf(:, k) + toc();
        m_ghf(:, :, k) = m;
        m_ghs(:, :, k) = ms;
        
        % RB-GHF and RB-GHS
        [Xi, w] = gh_sigmapoints(zeros(ds, 1), eye(ds), 3);        
        tic();
        [m, P, mp, Pp, Cp] = rbgf(ys(:, :, k), m0, P0, f, A, Q, g, B, R, is, iz, Xi, w);
        t_rbghf(k) = toc;
        
        tic();
        [ms, Ps] = rts_smooth(m, P, mp, Pp, Cp);
        t_rbghs(k) = t_rbghf(k) + toc();
        m_rbghf(:, :, k) = m;
        m_rbghs(:, :, k) = ms;
    end
    
    %% Particle filter/smoother
    if particle
        % Rao-Blackwellized bootstrap particle filter
        tic;
        [shat_rbpf, zhat_rbpf] = rbpf(ys(:, :, k), m0, P0, f, A, Q, g, B, R, is, iz, Jf);
        t_rbpf(:, k) = toc;
        m_rbpf(:, :, k) = [shat_rbpf; zhat_rbpf];

        % Rao-Blackwellized FFBSi particle smoother
        tic;
        [~, ~, sf, lwf, mzf, Pzf] = rbpf(ys(:, :, k), m0, P0, f, A, Q, g, B, R, is, iz, Jf);
        [shat_rbffbsi, zhat_rbffsi] = rbffbsi(ys(:, :, k), sf, lwf, mzf, Pzf, m0, P0, f, A, Q, g, B, R, is, iz, Js);
        t_rbffbsi(:, k) = toc;
        m_rbps(:, :, k) = [shat_rbffbsi; zhat_rbffsi];
    end
    
    %% Progress update
    fprintf(repmat(char(8), [1, length(str)]));
end

%% Mean execution time
dt_ukf = mean(t_ukf)/N;
dt_rbukf = mean(t_rbukf)/N;
dt_urtss = mean(t_urtss)/N;
dt_rburtss = mean(t_rburtss)/N;

dt_ghf = mean(t_ghf)/N;
dt_rbghf = mean(t_rbghf)/N;
dt_ghs = mean(t_ghs)/N;
dt_rbghs = mean(t_rbghs)/N;

dt_rbpf = mean(t_rbpf);
dt_rbffbsi = mean(t_rbffbsi);

%% Error, RMSE
e_ukf = xs - m_ukf;
e_rbukf = xs - m_rbukf;
e_urtss = xs - m_urtss;
e_rburtss = xs - m_rburtss;

e_ghf = xs - m_ghf;
e_rbghf = xs - m_rbghf;
e_ghs = xs - m_ghs;
e_rbghs = xs - m_rbghs;

e_rbpf = xs - m_rbpf;
e_rbps = xs - m_rbps;

trmse = @(e) squeeze(mean(sqrt(sum(e.^2, 1)), 2));
rmse_ukf = mean(trmse(e_ukf));
rmse_rb_ukf = mean(trmse(e_rbukf));
rmse_ghf = mean(trmse(e_ghf));
rmse_rb_ghf = mean(trmse(e_rbghf));

rmse_urtss = mean(trmse(e_urtss));
rmse_rburtss = mean(trmse(e_rburtss));
rmse_ghs = mean(trmse(e_ghs));
rmse_rbghs = mean(trmse(e_rbghs));

rmse_rbpf = mean(trmse(e_rbpf));
rmse_rbffbsi = mean(trmse(e_rbps));

%% Performance
fprintf('Filter \t\tTime\t\tRMSE\n');

fprintf('UKF: \t\t%.2e \t%.2e\n', dt_ukf, rmse_ukf);
fprintf('RB-UKF: \t%.2e\t%.2e\n', dt_rbukf, rmse_rb_ukf);
fprintf('GHF: \t\t%.2e\t%.2e\n', dt_ghf, rmse_ghf);
fprintf('RB-GHF: \t%.2e\t%.2e\n', dt_rbghf, rmse_rb_ghf);
fprintf('RB-PF: \t\t%.2e\t%.2e\n', dt_rbpf, rmse_rbpf);

fprintf('\n');
fprintf('Smoother \tTime\t\tRMSE\n');
fprintf('URTSS: \t\t%.2e\t%.2e\n', dt_urtss, rmse_urtss);
fprintf('RB-URTSS: \t%.2e\t%.2e\n', dt_rburtss, rmse_rburtss);
fprintf('GHS: \t\t%.2e\t%.2e\n', dt_ghs, rmse_ghs);
fprintf('RB-GHS: \t%.2e\t%.2e\n', dt_rbghs, rmse_rbghs);
fprintf('RB-FFBSi: \t%.2e\t%.2e\n', dt_rbffbsi, rmse_rbffbsi);

%% Visualize
k = 1;
t = (1:N)*Ts;
figure(1); clf();
figure(2); clf();
for i = 1:dx
    figure(1);
    subplot(dx, 1, i);
    plot(t, e_ukf(i, :, k)); hold on;
    plot(t, e_rbukf(i, :, k));
    plot(t, e_ghf(i, :, k));
    plot(t, e_rbghf(i, :, k));
    plot(t, e_rbpf(i, :, k));
    xlabel('t / s');
    legend('UKF', 'RB-UKF', 'GHF', 'RB-GHF', 'RB-PF');
    title(sprintf('Filter error, state #%d, MC run #%d', i, k));
    
    figure(2);
    subplot(dx, 1, i);
    plot(t, e_urtss(i, :, k)); hold on;
    plot(t, e_rburtss(i, :, k));
    plot(t, e_ghs(i, :, k));
    plot(t, e_rbghs(i, :, k));
    plot(t, e_rbps(i, :, k));
    xlabel('t / s');
    legend('URTSS', 'RB-URTSS', 'GHS', 'RB-GHS', 'RB-FFBSi');
    title(sprintf('Smoother error, state #%d, MC run #%d', i, k));
end

%% Export results
if export
    if exist('results/results.mat', 'file')
        load('results/results.mat');
    else
        % 5 parameter settings (dz), 5 filters/smoothers
        dzs = (2:2:10).';
        dt_filter = zeros(5, 5);
        rmse_filter = zeros(5, 5);
        dt_smoother = zeros(5, 5);
        rmse_smoother = zeros(5, 5);
    end
    
    dt_filter(dzs == dz, :) = [dt_ukf, dt_rbukf, dt_ghf, dt_rbghf, dt_rbpf];
    rmse_filter(dzs == dz, :) = [rmse_ukf, rmse_rb_ukf, rmse_ghf, rmse_rb_ghf, rmse_rbpf];
    dt_smoother(dzs == dz, :) = [dt_urtss, dt_rburtss, dt_ghs, dt_rbghs, dt_rbffbsi];
    rmse_smoother(dzs == dz, :) = [rmse_urtss, rmse_rburtss, rmse_ghs, rmse_rbghs, rmse_rbffbsi];
    
    save('results/results.mat', 'dzs', 'dt_filter', 'rmse_filter', 'dt_smoother', 'rmse_smoother');
end
