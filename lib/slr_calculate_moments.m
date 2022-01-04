function [my, Pyy, Pxy] = slr_calculate_moments(mx, Pxx, h, R, Xi, wm, wc)
% Calculate expectations for statistical linear regression
%
% Syntax:
%   [my, Pyy, Pxy] = SLR_CALCULATE_MOMENTS(mx, Pxx, h, R, Xi, wm, wc)
%
% In:
%   mx  Mean of the prior distribution (dx*1)
%   Pxx Covariance matrix of the prior distribution (dx*dx)
%   h   Nonlinear transformation, function handle @(x)
%   R   Covariance of the additive noise, matrix (dx*dx) or function handle
%       @(x)
%   Xi  Unit sigma-points
%   wm  Weights for calculating the mean
%   wc  Weights for calculating the covariance (optional, default: wm)
%
% Out:
%   my  Mean of the transformed variable w.r.t. q(x) (dy*1)
%   Pyy Covariance of the transformed variable w.r.t. q(x) (dy*dy)
%   Pxy Cross-covariance of the input and transformed variable w.r.t. q(x)
%       (dx*dy)
%
% Description:
%   Calculates the mean, covariance, and cross-covariance of the nonlinear
%   transformation
%
%       y = h(x) + r,
%
%   where r ~ N(0, R(x)) is additive noise, with respect to to the
%   density q(x) = N(x; mx, Pxx). The moments are calculated using
%   sigma-point integration.

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
    wm = wm(:);
    if nargin < 5 || isempty(wc)
        wc = wm;
    end
    
    % Make sure that R is a function
    if ~isa(R, 'function_handle')
        R = @(x) R;
    end
    
    % Dimensions
    M = size(Xi, 2);
    dx = size(mx, 1);
    dy = size(R(mx), 1);
    
    %% Sigma-point integration
    % Calculate sigma-points
    Lxx = chol(Pxx).';
    X = mx*ones(1, M) + Lxx*Xi;
    
    % Propagate sigma points
    Y = zeros(dy, M);
    for m = 1:M
        Y(:, m) = h(X(:, m));
    end
    
    % Calculate the mean
    my = Y*wm;
    
    % Calculate the covariances
    Pyy = zeros(dy, dy);
    Pxy = zeros(dx, dy);
    for m = 1:M
        Pyy = Pyy + wc(m)*((Y(:, m) - my)*(Y(:, m) - my)') + wc(m)*R(X(:, m));
        Pxy = Pxy + wc(m)*((X(:, m) - mx)*(Y(:, m) - my)');
    end
    Pyy = (Pyy + Pyy')/2;
end
