function [m, P, K, yp, S, lpy] = gf_update(mp, Pp, y, g, R, Xi, wm, wc)
% Sigma-point Gaussian filter measurement update for additive noise models
%
% Syntax:
%   [m, P, K, yp, S, lpy] = GF_UPDATE(mp, Pp, y, g, R, Xi, wm, wc)
%
% In:
%   mp  Predicted mean of the previous time step (Nx*1)
%   Pp  Predicted covariance of the previous time step (Nx*Nx)
%   y   Current measurement (Ny*1)
%   g   Nonlinear observation function, function handle @(x)
%   R   Measurement noise covariance, matrix (Ny*Ny) or function handle
%       @(x)
%   Xi  Unit sigma-points (Nx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   m   Updated mean (Nx*1)
%   P   Updated covariance (Nx*Nx)
%   K   Kalman gain (Nx*Ny)
%   yp  Predicted output (Ny*1)
%   S   Predicted output covariance (Ny*Ny)
%   lpy Marginal log-likelihood
%
% Description:
%   Calculates the Gaussian filter mesurement update for nonlinear
%   measurement models with additive Gaussian noise of the form
%   
%       y[n] = g(x[n]) + r[n],
%
%   where r[n] ~ N(0, R(x[n])).
%
% See also:
%   gf_predict

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
    narginchk(7, 8);
    if nargin < 8 || isempty(wc)
        wc = wm;
    end

    %% Update
    % Predict output
    [yp, S, Cxy] = slr_calculate_moments(mp, Pp, g, R, Xi, wm, wc);
    
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
