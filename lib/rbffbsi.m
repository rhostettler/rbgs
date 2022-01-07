function [shat, zhat] = rbffbsi(y, sf, lwf, mzf, Pzf, m0, P0, f, A, Q, g, B, R, is, iz, Js)
% Rao-Blackwellized forward-filtering backward-simulation smoother
% 
% Syntax:
%   [shat, zhat] = RBFFBSI(y, sf, lwf, mzf, Pzf, m0, P0, f, A, Q, g, B, R, is, iz, Js)
%
% In: 
%   y   Measurements (dy*N)
%   sf  Forward filter particles (cell array)
%   lwf Forward filter particle weights (cell array)
%   mzf Forward filter conditional means (cell array)
%   Pzf Forward filter conditional covariances (cell array)
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
%   Js  Number of particles (backward trajectories; default: 100)
%
% Out:
%   shat    MMSE estimate of the nonlinear states (posterior mean)
%   zhat    MMSE estimate of the linear states (posterior mean)
%
% Description:
%   Rao-Blackwellized forward-filtering backward-simulation particle 
%   smoother [1] for conditionally linear Gaussian state-space models
%   mixing models) of the form
%
%     x[n] = f(s[n-1]) + A(s[n-1])*z[n-1] + q[n]
%     y[n] = g(s[n]) + B(s[n])*z[n] + r[n],
%
%   with q[n] ~ N(0, Q(s[n-1])) and r[n] ~ N(0, R(s[n])) and where s[n] and
%   z[n] are the nonlinear and linear state components of x[n].
%
% References:
%   1. F. Lindsten, P. Bunch, S. Särkkä, T. Schön, S. Godsill,
%      "Rao-Blackwellized Particle Smoothers for Conditionally Linear
%      Gaussian Models", Journal of Selected Topics in Signal
%      Processing, vol. 10, no. 2, March 2016
%
% See also:
%   rbpf
    
%{
% Copyright (C) 2017-present Roland Hostettler
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
    narginchk(15, 16),
    if nargin < 16 || isempty(Js)
        Js = 100;
    end

    %% Preallocate
    N = size(sf, 2);
    ds = size(is, 1);
    dz = size(iz, 1);
    dy = size(y, 1);

    % MMSE for nonlinear states
    shat = zeros(ds, N);
    
    % Sufficient statistics
    lambda_hat = zeros(dz, Js);
    Omega_hat = zeros(dz, dz, Js);
    
    % Smoothed states and sufficient statistics
    sss = cell(1, N);
    lambdas = cell(1, N);
    Omegas = cell(1, N);
    
    %% Initialize smoother
    % Prepend NaN for n = 0
    y = [NaN*ones(dy, 1), y];
    
    % Resample filter particles
    s = sf{N};
    lw = lwf{N};
    alpha = catrand(lw, Js);
    ss = s(:, alpha);

    % Initialize sufficient statistics
    for j = 1:Js
        gj = g(ss(:, j));
        Bj = B(ss(:, j));
        Rj = R(ss(:, j));
        lambda_hat(:, j) = Bj'/Rj*(y(:, N) - gj);
        Omega_hat(:, :, j) = Bj'/Rj*Bj;
    end
    
    % MMSE
    shat(:, N) = mean(ss, 2);
    
    % Store
    sss{N} = ss;
    lambdas{N} = zeros(dz, Js);
    Omegas{N} = zeros(dz, dz, Js);
    
    %% Backward simulation
    for n = N-1:-1:1
        % Get filtering particles, weights, and condtional means and
        % covariances
        s = sf{n};
        lw = lwf{n};
        mz = mzf{n};
        Pz = Pzf{n};

        % Extend each backward trajectory...
        for j = 1:Js
            % 3a)-3c): Backward prediction
            [lv, Omega, lambda] = calculate_ancestor_weights(ss(:, j), lambda_hat(:, j), Omega_hat(:, :, j), s, lw, mz, Pz, f, A, Q, is, iz);

            % 3d)-3f): Extend the state trajectory (sample an ancestor
            % index and extend the trajectory)
            beta = catrand(lv, 1);
            ss(:, j) = s(:, beta);
            Omega(:, :, j) = Omega(:, :, beta);
            lambda(:, j) = lambda(:, beta);

            % 3g) Compute Omega_hat, lambda_hat
            gj = g(ss(:, j));
            Bj = B(ss(:, j));
            Rj = R(ss(:, j));
            lambda_hat(:, j) = lambda(:, j) + Bj'/Rj*(y(:, n)-gj);
            Omega_hat(:, :, j) = Omega(:, :, j) + Bj'/Rj*Bj;
        end
        
        % MMSE
        shat(:, n) = mean(ss, 2);
        
        % Store
        sss{n} = ss;
        lambdas{n} = lambda;
        Omegas{n} = Omega;
    end
    
    %% Smooth linear states
    zhat = crts_smooth(y, sss, lambdas, Omegas, m0, P0, f, A, Q, g, B, R, is, iz);
    
    %% Remove initial mean
    shat = shat(:, 2:N);
    zhat = zhat(:, 2:N);
end

%% Calculates the ancestor weights and statistics
function [lv, Omega, lambda] = calculate_ancestor_weights(ss, lambda_hat, Omega_hat, s, lw, mz, Pz, f, A, Q, is, iz)
    %% Preliminaries
    [ds, Jf] = size(s);
    dz = size(mz, 1);
    Is = eye(ds);
    Iz = eye(dz);
    Ix = eye(ds+dz);

    %% Calculations
    Omega = zeros(dz, dz, Jf);
    lambda = zeros(dz, Jf);
    lv = zeros(1, Jf);            
    for j = 1:Jf
        % 3a)
        % Evaluate dynamic functions and matrices and get nonlinear
        % subfunctions/matrices
        % original article's notation)
        fj = f(s(:, j));
        Aj = A(s(:, j));
        Qj = Q(s(:, j));
        
        fs = fj(is);
        As = Aj(is, :);
        Qs = Qj(is, is);
        Gs = zeros(ds, ds + dz);        % TODO: Rename to avoid naming conflicts
        Gs(:, is) = Is;
        
        fz = fj(iz);
        Az = Aj(iz, :);
        Fs = zeros(dz, ds + dz);        % TODO: Rename to avoid naming conflicts
        Fs(:, iz) = Iz;

        % Gram-Schmidt orthogonalization
        Qbar = Ix - Gs'/Qs*Gs;
        fbar = fz + Fs*Gs'/Qs*(ss - fs);
        Abar = Az - Fs*Gs'/Qs*As;

        % Actual calculation of the backward statistics
        mt = lambda_hat - Omega_hat*fbar;
        Mt = Qbar*Fs'*Omega_hat*Fs*Qbar + Ix;
        Psi = Qbar/Mt*Qbar;
        tau = ( ...
            (ss-fs)'/Qs*(ss-fs) ...
            ... %wnorm(ss-fs, Qs\Is) ...
            + fbar'*Omega_hat*fbar ...
            - 2*lambda_hat'*fbar ...
            - (Fs'*mt)'*Psi*(Fs'*mt) ...
        );

        logZ = -1/2*(log(det(Qs)) + log(det(Mt)) + tau);
        K = Abar'*(Iz - Omega_hat*Fs*Psi*Fs');
        Omega(:, :, j) = K*Omega_hat*Abar + As'/Qs*As;
        lambda(:, j) = K*mt + As'/Qs*(ss-fs);

        % 3b)
        zbar = mz(:, j);
        Gamma = chol(Pz(:, :, j)).';
        Lambda = Gamma'*Omega(:, :, j)*Gamma + Iz;
        epsilon = Gamma'*(lambda(:, j) - Omega(:, :, j)*zbar);
        eta = ( ...
            zbar'*Omega(:, :, j)*zbar ...
            - 2*lambda(:, j)'*zbar ...
            - epsilon'/Lambda*epsilon ...
        );

        % 3c)
        lv(j) = lw(j) + logZ - 1/2*log(det(Lambda)) - 1/2*eta;
    end

    % 3c) Normalize the weights
    v = exp(lv-max(lv));
    v = v/sum(v);
    lv = log(v);
end

%% Conditional Rauch-Tung-Striebel smoother for the linear states
function [zhat, mzss, Pzss] = crts_smooth(y, sss, lambdas, Omegas, m0, P0, f, A, Q, g, B, R, is, iz)
    %% Preallocate
    N = size(y, 2);
    Js = size(sss{1}, 2);
    dz = size(iz, 1);
    Iz = eye(dz);
    
    zhat = zeros(dz, N);
    mzs = zeros(dz, Js);
    Pzs = zeros(dz, dz, Js);
    
    mzss = cell(1, N);
    Pzss = cell(1, N);
    
    %% Initialize
    ss = sss{1};
    K = P0(iz, is)/P0(is, is);
    mz = m0(iz)*ones(1, Js) + K*(ss - m0(is)*ones(1, Js));
    Pz = P0(iz, iz) - K*P0(is, is)*K';
    Pz = (Pz + Pz')/2;
    Pz = repmat(Pz, [1, 1, Js]);

    %% Process all samples
    % N.B.: Do not process n = 1 (initialization sample)
    for n = 2:N
        sp = ss;
        ss = sss{n};
        lambda = lambdas{n};
        Omega = Omegas{n};

        %% Process all trajectories
        for j = 1:Js
            %% KF Prediction
            % Move from n-1 to n
            fj = f(sp(:, j));
            Aj = A(sp(:, j));
            Qj = Q(sp(:, j));
            fz = fj(iz);
            Az = Aj(iz, :);
            Qz = Qj(iz, iz);
            
            % Prediction
            mzp = fz + Az*mz(:, j);
            Pzp = Az*Pz(:, :, j)*Az' + Qz;

            %% KF Measurement Update
            % Update at n
            gj = g(ss(:, j));
            Bj = B(ss(:, j));
            Rj = R(ss(:, j));
            S = Bj*Pzp*Bj' + Rj;
            K = (Pzp*Bj')/S;
            mz(:, j) = mzp + K*(y(:, n) - gj - Bj*mzp);
            Pz(:, :, j) = Pzp - K*S*K';

            %% Smoothed Means & Covariances
            Pzs(:, :, j) = (Pz(:, :, j)\Iz + Omega(:, :, j))\Iz;
            mzs(:, j) = Pzs(:, :, j)*(Pz(:, :, j)\mz(:, j) + lambda(:, j));
        end
        
        % Store
        zhat(:, n) = mean(mzs, 2);
        mzss{n} = mzs;
        Pzss{n} = Pzs;
    end
end
