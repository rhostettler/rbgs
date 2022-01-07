function [ms, Ps, mps, Pps, Cps, lpys] = rbgf(y, m0, P0, f, A, Q, g, B, R, in, il, Xi, wm, wc)
% Rao-Blackwellized Gaussian filter for mixed linear/nonlinear SSMs
%
% Syntax:
%   [m, P, mp, Pp, Cp, lpy] = RBGF(y, m0, P0, f, A, Q, g, B, R, in, il, Xi, wm, wc)
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
%   in  Vector of indices to the nonlinear components s of x (ds*1)
%   il  Vector of indices to the linear components z of x (dz*1)
%   Xi  Unit sigma-points (Nx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   m   Filtered means (dx*N)
%   P   Filtered covariances (dx*dx*N)
%   mp  Predicted means (dx*N)
%   Pp  Predicted covariances (dx*dx*N)
%   Cp  Predicted cross-covariances (dx*dx*N)
%   lpy Marginal log-likelihoods (1*N)
%
% Description:
%   Rao-Blackwellized Gaussian filter for mixing linear/nonlinear
%   state-space models of the form
%
%       x[n] = f(s[n-1]) + A(s[n-1]) z[n-1] + q[n],
%       y[n] = g(s[n]) + B(s[n]) z[n] + r[n],
%
%   where q[n] ~ N(0, Q(s[n-1])), and r[n] ~ N(0, R(s[n])).
%
% See also:
%   rbgf_predict, rbgf_update

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
    narginchk(13, 14);
    if nargin < 14 || isempty(wc)
        wc = wm;
    end

    %% Preallocate
    N = size(y, 2);
    dx = size(m0, 1);
    ms = zeros(dx, N);
    mps = ms;
    Ps = zeros(dx, dx, N);
    Pps = Ps;
    Cps = Ps;
    lpys = zeros(1, N);
    
    %% Process data
    m = m0;
    P = P0;
    for n = 1:N
        [mp, Pp, Cp] = rbgf_predict(m, P, f, A, Q, in, il, Xi, wm, wc);
        mps(:, n) = mp;
        Pps(:, :, n) = Pp;
        Cps(:, :, n) = Cp;
        
        [m, P, ~, ~, ~, lpy] = rbgf_update(y(:, n), mp, Pp, g, B, R, in, il, Xi, wm, wc);
        ms(:, n) = m;
        Ps(:, :, n) = P;
        lpys(:, n) = lpy;
    end
end
