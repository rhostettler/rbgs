function [my, Pyy, Pxy] = rbslr_calculate_moments(mx, Pxx, h, H, R, in, il, Xi, wm, wc)
% Rao-Blackwellized calculation of the expectations for SLR
%
% Syntax:
%   [my, Pyy, Pxy] = RBSLR_CALCULATE_MOMENTS(mx, Pxx, h, H, R, in, il, Xi, wm, wc)
%
% In:
%   mx  Mean of the prior distribtuion (dx*1)
%   Pxx Covariance of the prior distribution (dx*dx)
%   h   Nonlinear function of the nonlinear states, function handle @(s)
%   H   Matrix of linear transformation of the linear states, matrix
%       (dy*dz) or function handle @(s)
%   R   Additive noise covariance, matrix (dy*dy) or function handle @(s)
%   in  Vector of indices to the nonlinear components s of x (ds*1)
%   il  Vector of indices to the linear components z of x (dz*1)
%   Xi  Unit sigma-points (ds*M)
%   wm  Weights for calculating the mean (1*M)
%   wc  Weights for calculating the covariances (1*M; optional, default:
%       wm)
%
% Out:
%   my  Mean of the transformed variable w.r.t. q(x) (dy*1)
%   Pyy  Covariance of the transformed variable w.r.t. q(x) (dy*dy)
%   Pxy Cross-covariance of the input and transformed variable w.r.t. q(x)
%       (dx*dy)
%
% Description:
%   Rao-Blackwellized calculation of the mean, covariance, and
%   cross-covariance of the conditionally affine transformation
%
%       y = h(s) + H(s)*z + r,
%
%   where v ~ N(0, R(s)), with respect to the prior density q(x) =
%   N(x; mx, Pxx) and where s = [x]_in and z = [x]_il.

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
    
    % Ensure that H and Cv are functions of xn (if constant matrices are
    % given)
    if ~isa(H, 'function_handle')
        H = @(xn) H;
    end
    if ~isa(R, 'function_handle')
        R = @(xn) R;
    end

    % Dimensions
    M = size(Xi, 2);
    dx = size(mx, 1);
    ds = length(in);
    dz = length(il);
    dy = size(R(mx), 1);

    %% Calculate moments
    ms = mx(in);
    mz = mx(il);
    Pss = Pxx(in, in);
    Pzz = Pxx(il, il);
    Psz = Pxx(in, il);

    % Sigma-points
    Lss = chol(Pss).';
    Ss = ms*ones(1, M) + Lss*Xi;
    
    % Orthogonalization
    K = Psz'/Pss;
    z_tilde = mz*ones(1, M) + K*(Ss - ms*ones(1, M));
    Pzz_tilde = Pzz - K*Pss*K';
    
    % Function evaluations
    hs = zeros(dy, M);
    Hs = zeros(dy, dz, M);
    Rs = zeros(dy, dy, M);
    Ys = zeros(dy, M);
    for m = 1:M
        hs(:, m) = h(Ss(:, m));
        Hs(:, :, m) = H(Ss(:, m));
        Rs(:, :, m) = R(Ss(:, m));
        Ys(:, m) = hs(:, m) + Hs(:, :, m)*z_tilde(:, m);
    end
    
    % Mean
    my = Ys*wm(:);
    
    % Covariances
    Pyy = zeros(dy, dy);
    Pxy = zeros(dx, dy);
    sumH = zeros(dz, dy);
    for m = 1:M
        Pyy = Pyy + wc(m)*( ...
            ((Ys(:, m) - my)*(Ys(:, m) - my)') ...
            + Hs(:, :, m)*Pzz_tilde*Hs(:, :, m)' ...
            + Rs(:, :, m) ...
        );
        Pxy(in, :) = Pxy(in, :) + wc(m)*((Ss(:, m) - ms)*(Ys(:, m) - my)');
        sumH = sumH + wc(m)*Hs(:, :, m)';
    end
    Pxy(il, :) = K*Pxy(in, :) + Pzz_tilde*sumH;
end
