function [ms, Ps, mps, Pps, Cps, lpys] = gf(m0, P0, y, f, Q, g, R, Xi, wm, wc)
% Sigma-point Gaussian filter for nonlinear, additive noise models
%
% Syntax:
%   [m, P, mp, Pp, Cp, lpy] = GF(m0, P0, y, f, Q, g, R, Xi, wm, wc)
%
% In:
%   m0  Initial mean (Nx*1)
%   P0  Initial covariance (Nx*Nx)
%   y   Measurements (Ny*N)
%   f   Dynamic function, function handle of the form @(x)
%   Q   Process noise covariance, matrix (Nx*Nx) or function handle of the
%       form @(x)
%   g   Nonlinear observation function, function handle @(x)
%   R   Measurement noise covariance, matrix (Ny*Ny) or function handle
%       @(x)
%   Xi  Unit sigma-points (Nx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   m   Filtered means (Nx*N)
%   P   Filtered covariances (Nx*Nx*N)
%   mp  Predicted means (Nx*N)
%   Pp  Predicted covariances (Nx*Nx*N)
%   Cp  Predicted cross-covariances (Nx*Nx*N)
%   lpy Marginal log-likelihoods (1*N)
%
% Description:
%   Gaussian filter for nonlinear state-space models with additive Gaussian
%   noise of the form
%
%       x[n] = f(x[n-1]) + q[n],
%       y[n] = g(x[n]) + r[n],
%
%   where q[n] ~ N(0, Q(x[n-1])), and r[n] ~ N(0, R(x[n])). The function
%   loops over a whole set of data.
%
% See also:
%   gf_predict, gf_update

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
    narginchk(9, 10);
    if nargin < 10 || isempty(wc)
        wc = wm;
    end
    
    %% Preallocate
    N = size(y, 2);
    Nx = size(m0, 1);
    ms = zeros(Nx, N);
    mps = ms;
    Ps = zeros(Nx, Nx, N);
    Pps = Ps;
    Cps = Ps;
    lpys = zeros(1, N);
    
    %% Process data
    m = m0;
    P = P0;
    for n = 1:N
        [mp, Pp, Cp] = gf_predict(m, P, f, Q, Xi, wm, wc);
        mps(:, n) = mp;
        Pps(:, :, n) = Pp;
        Cps(:, :, n) = Cp;
        
        [m, P, ~, ~, ~, lpy] = gf_update(mp, Pp, y(:, n), g, R, Xi, wm, wc);
        ms(:, n) = m;
        Ps(:, :, n) = P;
        lpys(:, n) = lpy;
    end
end
