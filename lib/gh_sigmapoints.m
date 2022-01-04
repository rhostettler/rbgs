function [Xi, wm] = gh_sigmapoints(m, P, p)
% Gauss-Hermite cubature sigma-points
% 
% Syntax:
%   [Xi, wm] = GH_SIGMAPOINTS(m, P, p)
%
% In:
%   m   Prior mean (default: 0).
%   P   Prior covariance (default: 1).
%   p   Polynomial order (default: 3).
% 
% Out:
%   Xi  Sigma points.
%   wm  Weights.
%   
% Description:
%   Implements the multi-dimensional Gauss-Hermite cubature rules.
% 
% See also:
%   ut_sigmapoints

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
    narginchk(0, 3);
    if nargin < 1 || isempty(m)
        m = 0;
    end
    dx = size(m, 1);
    if nargin < 2 || isempty(P)
        P = eye(dx);
    end
    if nargin < 3 || isempty(p)
        p = 3;
    end
    if size(m, 1) ~= size(P, 1)
        error('Mean and covariance dimensions mismatch.');
    end
    
    %% Calculate unit sigma-points
    dx = size(m, 1);
    [xi, w] = calculate_roots(p);
    Xi = zeros(dx, p^dx);
    wm = zeros(1, p^dx);

    % The goal is to create the dxp^N matrix of sigma points Xi and
    % the 1xp^N weight vector. The matrix has the following form:
    %        _                                                                 _
    %       | xi(1) xi(1) ... xi(1) xi(2) xi(2) ... xi(2) xi(p) xi(p) ... xi(p) |
    %       | xi(1) xi(1)     xi(p) xi(1) xi(1) ... xi(p) xi(1) xi(1) ... xi(p) |
    %  Xi = | ...   ...       ...   ...   ...       ...   ...   ...      ...    |
    %       |_xi(1) xi(2) ... xi(p) xi(1) xi(2) ... xi(p) xi(1) xi(2) ... xi(p)_|
    %
    % The weights are constructed in the same way and in the end
    % the product of all the weights is taken.
                        
    for n = 1:dx
        % First we create the basic repetitive pattern, i.e. how
        % many times x(1) should be repeated in this row.
        % Then, we repeat that pattern p^(n-1) times to build a
        % full row and put everything together.
        rxi = ones(p^(dx-n), 1)*xi(:).';                
        rxi = rxi(:)*ones(1, p^(n-1));
        Xi(n, :) = rxi(:).';

        % Calculate the weights
        rw = ones(p^(dx-n), 1)*w(:).';
        rw = rw(:)*ones(1, p^(n-1));
        wm(n, :) = rw(:).';
    end
    wm = prod(wm, 1);
    
    %% Calculate the non-unit sigma-points
    Xi = m*ones(1, size(Xi, 2)) + chol(P).'*Xi;
end       
     
%% Calculates the Roots of the Gauss-Hermite Polynomial of Order p
function [xi, w] = calculate_roots(p)
    switch p
        case 0
            xi = [];
        case 1
            xi = 0;
        case 2
            xi = [-1, 1].';
        case 3
            xi = [-sqrt(3), 0, sqrt(3)].';
        case 4
            xi = [
                -sqrt((6+sqrt(24))/2);
                -sqrt((6-sqrt(24))/2);
                 sqrt((6-sqrt(24))/2);
                 sqrt((6+sqrt(24))/2);
            ];
        otherwise
            error('Not implemented.');
    end
    w = factorial(p)./(p^2*calculate_hermite_polynomial(p-1, xi).^2);
end

%% Gauss-Hermite Polynomial
function y = calculate_hermite_polynomial(p, x)
    switch p
        case 0
            y = 1;
        case 1
            y = x;
        case 2
            y = x.^2 - 1;
        case 3
            y = x.^3 - 3*x;
        case 4
            y = x.^4 - 6*x.^2 + 3;
        otherwise
            y = ( ...
                x.*calculate_hermite_polynomial(p-1, x) ...
                - (p-1)*calculate_hermite_polynomial(p-2, x) ...
            );
    end
end
