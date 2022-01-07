function [shat, zhat, sf, lwf, mzf, Pzf] = rbpf(y, m0, P0, f, A, Q, g, B, R, is, iz, J)
% Rao-Blackwellized bootstrap particle filter for CLGSS models
% 
% Syntax:
%   [shat, zhat] = RBPF(y, m0, P0, f, A, Q, g, B, R, is, iz, J)
% 
% In:
%   y   Measurements (dy*N)
%   m0  Initial mean (dx*1)
%   P0  Initial covariance (dx*dx)
%   f   Nonlinear dynamic function, function handle of the form @(s)
%   A   Linear state dynamics matrix, matrix (dx*dx) or function handle of
%       the form @(s)
%   Q   Process noise covariance, matrix (dx*dx) or function handle of the
%       form @(s)
%   g   Nonlinear observation function, function handle @(s)
%   B   Linear observation matrix, matrix (dy*dz) or function handle @(s)
%   R   Measurement noise covariance, matrix (dy*dy) or function handle
%       @(s)
%   is  Vector of indices to the nonlinear components s of x (ds*1)
%   iz  Vector of indices to the linear components z of x (dz*1)
%   J   Number of particles (default: 100)
%
% Out:
%   shat    MMSE estimate of the nonlinear states (posterior mean)
%   zhat    MMSE estimate of the linear states (posterior mean)
%   sf      Filter particles (cell array)
%   lwf     Filter particle weights (cell array)
%   mzf     Filter conditional means (cell array)
%   Pzf     Filter conditional covariances (cell array)
% 
% Description:
%   Rao-Blackwellized bootstrap particle filter (RBPF) for conditionally
%   linear Gaussian state-space models (mixing models) of the form
%
%     x[n] = f(s[n-1]) + A(s[n-1])*z[n-1] + q[n]
%     y[n] = g(s[n]) + B(s[n])*z[n] + r[n],
%
%   with q[n] ~ N(0, Q(s[n-1])) and r[n] ~ N(0, R(s[n])) and where s[n] and
%   z[n] are the nonlinear and linear state components of x[n].
% 
% References:
%   1. T. Schön, F. Gustafsson, and P.-J. Nordlund, “Marginalized particle 
%      filters for mixed linear/nonlinear state-space models,” IEEE 
%      Transactions on Signal Processing, vol. 53, no. 7, pp. 2279–2289, 
%      July 2005.
%   2. F. Lindsten, P. Bunch, S. Särkkä, T. B. Schön, and S. J. Godsill, 
%      “Rao–Blackwellized particle smoothers for conditionally linear 
%      Gaussian models,” IEEE Journal of Selected Topics in Signal 
%      Processing, vol. 10, no. 2, pp. 353–365, March 2016.
%
% See also:
%   rbffbsi

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
    
    %% Defaults
    narginchk(11, 12);
    if nargin < 12 || isempty(J)
        J = 100;
    end
    
    %% Preallocate
    % State dimensions
    ds = size(is, 1);
    dz = size(iz, 1);
    [dy, N] = size(y);
    
    % Preallocate output variables
    sf = cell(1, N+1);
    lwf = cell(1, N+1);
    mzf = cell(1, N+1);
    Pzf = cell(1, N+1);
    shat = zeros(ds, N);
    zhat = zeros(dz, N);

    %% Initialize
    % Draw initial samples for the nonlinear states
    s = m0(is)*ones(1, J) + chol(P0(is, is)).'*randn(ds, J);
    lw = log(1/J)*ones(1, J);
    
    % Conditionally linear states
    K = P0(iz, is)/P0(is, is);
    mz = m0(iz)*ones(1, J) + K*(s - m0(is)*ones(1, J));
    Pz = P0(iz, iz) - K*P0(is, is)*K';
    Pz = (Pz + Pz')/2;
    Pz = repmat(Pz, [1, 1, J]);
    
    % Store
    sf{1} = s;
    lwf{1} = lw;
    mzf{1} = mz;
    Pzf{1} = Pz;
    
    %% Process data
    % Expand data dimensions, prepend NaN for zero measurement
    y = [NaN*ones(dy, 1), y];
    N = N+1;
    for n = 2:N
        %% Time update
        % Resample and draw particles
        [sp, lw, s, mz, Pz] = sample(s, lw, mz, Pz, f, A, Q, is, iz);
        
        % KF prediction for conditionally linear states
        [mzp, Pzp] = kf_predict(sp, s, mz, Pz, f, A, Q, is, iz);
        
        %% Measurement update
        % Calculate and normalize weights
        s = sp;
        lw = calculate_weights(y(:, n), s, lw, mzp, Pzp, g, B, R);
        w = exp(lw);
        if any(~isfinite(w))
            warning('NaN/Inf in particle weights.');
        end
                
        % Measurement update for conditionally linear states
        [mz, Pz] = kf_update(y(:, n), s, mzp, Pzp, g, B, R);
        
        %% Point estimates
        % MMSE
        shat(:, n-1) = s*w';
        zhat(:, n-1) = mz*w';

        %% Store
        sf{n} = s;
        lwf{n} = lw;
        mzf{n} = mz;
        Pzf{n} = Pz;
    end
end

%% Sampling function
function [sp, lw, s, mz, Pz] = sample(s, lw, mz, Pz, f, A, Q, is, iz)
    [ds, J] = size(s);
    
    %% Resampling
    w = exp(lw);
    w = w/sum(w);
    Jess = 1/sum(w.^2);
    r = (Jess < J/3);
    if r
        % ESS is too low: Sample according to the categorical distribution
        % defined by the sample weights w, alpha ~ Cat{w}
        alpha = catrand(lw, J);
        s = s(:, alpha);
        lw = log(1/J)*ones(1, J);
        mz = mz(:, alpha);
        Pz = Pz(:, :, alpha);
    end   

    %% Sampling (bootstrap)
    sp = zeros(ds, J);
    for j = 1:J
        fj = f(s(:, j));
        Aj = A(s(:, j));
        Qj = Q(s(:, j));
        msp = fj(is) + Aj(is, :)*mz(:, j);
        Qsp = Aj(is, :)*Pz(:, :, j)*Aj(is, :)' + Qj(is, is);
        
        sp(:, j) = msp + chol(Qsp).'*randn(ds, 1);
    end
end

%% Kalman filter prediction
function [mzp, Pzp] = kf_predict(sp, s, mz, Pz, f, A, Q, is, iz)
    %% Preallocate
    [dz, J] = size(mz);
    mzp = zeros(dz, J);
    Pzp = zeros(dz, dz, J);
    
    %% Propagate
    for j = 1:J
        % Evaluate the model at the current particle
        fj = f(s(:, j));
        Aj = A(s(:, j));
        Qj = Q(s(:, j));
        
        % Nonlinear function
        % s[n] = fs(s[n-1]) + As(s[n-1])*z[n-1] + qs[n]
        fs = fj(is);
        As = Aj(is, :);
        Qs = Qj(is, is);

        % Linear function
        % z[n] = fz(s[n-1]) + Az(s[n-1])*z[n-1] + qz[n]
        fz = fj(iz);
        Az = Aj(iz, :);
        Qz = Qj(iz, iz);
        Qzs = Qj(iz, is);

        % KF prediction
        M = As*Pz(:, :, j)*As' + Qs;
        L = (Az*Pz(:, :, j)*As' + Qzs)/M;
        mzp(:, j) = fz + Az*mz(:, j) + L*(sp(:, j) - fs - As*mz(:, j));
        Pzp(:, :, j) = Az*Pz(:, :, j)*Az' + Qz - L*M*L';
    end
end

%% Importance weights
function lw = calculate_weights(y, s, lw, mzp, Pzp, g, B, R)
    % Calculate incremental weights (likelihood)
    J = size(s, 2);
    lv = zeros(1, J);
    for j = 1:J
        gj = g(s(:, j));
        Bj = B(s(:, j));
        my = gj + Bj*mzp(:, j);
        Pyy = R(s(:, j)) + Bj*Pzp(:, :, j)*Bj';
        lv(j) = logmvnpdf(y.', my.', Pyy).';
    end
    
    % Non-normalized weight: Marginal likelihood + trajectory weight
    %
    % N.B.:
    % * If resampling has taken place, then lw is log(1/J)
    % * If no resampling has taken place, then lw is whatever it was from
    %   the previous iteration
    lw = lw + lv;    
    
    % Scale and normalize
    lw = lw-max(lw);
    w = exp(lw);
    w = w/sum(w);
    lw = log(w);
end

%% Kalman filter measurement update
function [mz, Pz] = kf_update(y, s, mzp, Pzp, g, B, R)
    [dz, J] = size(mzp);
    mz = zeros(dz, J);
    Pz = zeros([dz, dz, J]);    
    for j = 1:J
        gj = g(s(:, j));
        Bj = B(s(:, j));
        Rj = R(s(:, j));

        M = Bj*Pzp(:, :, j)*Bj' + Rj;
        K = Pzp(:, :, j)*Bj'/M;
        mz(:, j) = mzp(:, j) + K*(y - gj - Bj*mzp(:, j));
        Pz(:, :, j) = Pzp(:, :, j) - K*M*K';
    end
end
