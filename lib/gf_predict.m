function [mp, Pp, Cp] = gf_predict(m, P, f, Q, Xi, wm, wc)
% Gaussian filter prediction for additive noise model
%
% Syntax:
%   [mp, Pp, Cp] = GF_PREDICT(m, P, f, Q, Xi, wm, wc)
%
% In:
%   m   Mean of previous time step (Nx*1)
%   P   Covariance of previous time step (Nx*Nx)
%   f   Dynamic function, function handle of the form @(x)
%   Q   Process noise covariance, matrix (Nx*Nx) or function handle of the
%       form @(x)
%   Xi  Unit sigma-points (Nx*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   mp  Predicted mean (Nx*1)
%   Pp  Predicted covariance (Nx*Nx)
%   Cp  Predicted cross-covariance (Nx*Nx)
%
% Description:
%   Gaussian filter prediction for nonlinear additive noise models with 
%   state dynamics of the form
%
%       x[n] = f(x[n-1]) + q[n]
%
%   where q[n] ~ N(0, Q(x[n-1])).
%
% See also:
%   gf_update

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
    narginchk(6, 7);
    if nargin < 7 || isempty(wc)
        wc = wm;
    end

    %% Predict
    [mp, Pp, Cp] = slr_calculate_moments(m, P, f, Q, Xi, wm, wc);
end
