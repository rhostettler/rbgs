function [Xi, wm, wc] = ut_sigmapoints(m, P, alpha, beta, kappa)
% Unscented transform (UT) (and cubature) sigma-points
% 
% Syntax:
%   [Xi, wm, wc] = UT_SIGMAPOINTS(m, P, alpha, beta, kappa)
% 
% In:
%   m       Prior mean (default: 0)
%   P       Prior covariance (default: eye(dx))
%   alpha   UT spread parameter (default: 1)
%   beta    UT prior knowledge parameter (default: 0)
%   kappa   Secondary UT spread parameter (default: 1)
%
% Out:
%   Xi      Sigma points.
%   wm      Mean weights.
%   wc      Covariance weights.
% 
% Description:
%   Calculation of the sigma-points according to the unscented transform 
%   (UT) [1]. Note that setting the parameters as alpha = 1, beta = 0, 
%   kappa = 0 yields sigma-points according the cubature integration rule
%   [2].
%
% References:
%   1. Wan, E. A. and Van der Merwe, R. "The unscented Kalman filter", 
%      in Kalman Filtering and Neural Networks, Wiley, 2001
%
%   2. Arasaratnam, I. and Haykin, S. "Cubature Kalman filters", IEEE 
%      Transactions on Automatic Control, 54(6), 1254?1269, 2009
%
% See also:
%   gh_sigmapoints
    
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
    narginchk(0, 5);
    if nargin < 1 || isempty(m)
        m = 1;
    end
    dx = size(m, 1);
    if nargin < 2 || isempty(P)
        P = eye(dx);
    end
    if nargin < 3 || isempty(alpha)
        % Spread
        alpha = 1;
    end
    if nargin < 4 || isempty(beta)
        % Prior knowledge
        beta = 0;
    end
    if nargin < 5 || isempty(kappa)
        % Secondary spread
        kappa = 1;
    end
    if size(m, 1) ~= size(P, 1)
        error('Mean and covariance dimensions mismatch.');
    end
            
    %% Calculate the sigma-points
    lambda = alpha^2*(dx + kappa) - dx;
    wm = 1/(2*(dx + lambda))*ones(1, 2*dx + 1);
    wm(1) = lambda/(dx + lambda);
    wc = wm;
    wc(1) = lambda/(dx + lambda) + (1 - alpha^2 + beta);
    LP = sqrt(dx + lambda)*chol(P).';
    Xi = [m, m*ones(1, dx) + LP, m*ones(1, dx) - LP];
end
