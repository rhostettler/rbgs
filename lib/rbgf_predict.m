function [mp, Pp, Cp] = rbgf_predict(m, P, f, A, Q, in, il, Xi, wm, wc)
% Rao-Blackwellized Gaussian filter prediction for mixing CLGSS
%
% Syntax:
%   [mp, Pp, Cp] = RBGF_PREDICT(m, P, f, A, Q, in, il, Xi, wm, wc)
%
% In:
%   m   Mean of previous time step (dx*1)
%   P   Covariance of previous time step (dx*dx)
%   f   Nonlinear dynamic function, function handle of the form @(s)
%   A   Linear state dynamics matrix, matrix (dx*dx) or function handle of
%       the form @(s)
%   Q   Process noise covariance, matrix (dx*dx) or function handle of the
%       form @(s)
%   in  Vector of indices to the nonlinear components s of x (ds*1)
%   il  Vector of indices to the linear components z of x (dz*1)
%   Xi  Unit sigma-points (dx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   mp  Predicted mean (dx*1)
%   Pp  Predicted covariance (dx*dx)
%   Cp  Predicted cross-covariance (dx*dx)
%
% Description:
%   Rao-Blackwellized Gaussian filter prediction for mixing conditionally
%   linear Gaussian state-space models of the form
%
%       x[n] = f(s[n-1]) + A(s[n-1]) z[n-1] + q[n]
%
%   where q[n] ~ N(0, Q(s[n-1])).
%
% See also:
%   rbgf, rbgf_update

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
    
    %% Prediction
    [mp, Pp, Cp] = rbslr_calculate_moments(m, P, f, A, Q, in, il, Xi, wm, wc);
end
