function [m, P, K, yp, S, lpy] = rbgf_update(y, mp, Pp, g, B, R, in, il, Xi, wm, wc)
% Rao-Blackwellized Gaussian filter measurement update for CGLSS
%
% Syntax:
%   [m, P, K, yp, S, lpy] = RBGF_UPDATE(y, mp, Pp, g, B, R, in, il, Xi, wm, wc)
%
% In:
%   y   Current measurement (dy*1)
%   mp  Predicted mean of the previous time step (dx*1)
%   Pp  Predicted covariance of the previous time step (dx*dx)
%   g   Nonlinear observation function, function handle @(s)
%   B   Linear observation matrix, matrix (dy*dz) or function handle @(s)
%   R   Measurement noise covariance, matrix (dy*dy) or function handle
%       @(s)
%   in  Vector of indices to the nonlinear components s of x (ds*1)
%   il  Vector of indices to the linear components z of x (dz*1)
%   Xi  Unit sigma-points (dx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   m   Updated mean (dx*1)
%   P   Updated covariance (dx*dx)
%   K   Kalman gain (dx*dy)
%   yp  Predicted output (dy*1)
%   S   Predicted output covariance (dy*dy)
%   lpy Marginal log-likelihood
%
% Description:
%   Calculates the Rao-Blackwellized Gaussian filter mesurement update for 
%   conditionally linear observation models of the form
%   
%       y[n] = g(s[n]) + B(s[n]) z[n] + r[n],
%
%   where r[n] ~ N(0, R(s[n])).
%
% See also:
%   rbgf, rbgf_predict

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
    narginchk(10, 11);
    if nargin < 11 || isempty(wc)
        wc = wm;
    end
    
    %% Update
    % Predict output
    [yp, S, Cxy] = rbslr_calculate_moments(mp, Pp, g, B, R, in, il, Xi, wm, wc);
    
    % Mesurement update
    K = Cxy/S;
    v = y - yp;
    m = mp + K*v;
    P = Pp - K*S*K';
    P = (P+P')/2;

    % Calculate marginal log-likelihood
    if nargout == 6
        lpy = logmvnpdf(y, yp, S);
    end
end
