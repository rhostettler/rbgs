function [ms, Ps] = rts_smooth(m, P, mp, Pp, Cp)
% Rauch-Tung-Striebel smoother
%
% Syntax:
%   [ms, Ps] = RTS_SMOOTH(m, P, mp, Pp, Cp)
%
% In:
%   m   Filtered means (dx*N)
%   P	Filtered covariances (dx*dx*N)
%   mp  Predicted means (dx*N)
%   Pp  Predicted covariances (dx*dx*N)
%   Cp  Predicted cross-covariances (dx*dx*N)
%
% Out:
%   m   Smoothed means (dx*N)
%   P   Smoothed covariances (dx*dx*N)
%
% Description:
%   Rauch-Tung Striebel smoothing algorithm that refines the state
%   estimates obtained from a Kalman filter.
%
% See also:
%   gf

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
    narginchk(5, 5);
    
    % Preallocate
    N = size(m, 2);
    ms = zeros(size(m));
    Ps = zeros(size(P));
    
    %% Smoothing
    % Initialize
    ms(:, N) = m(:, N);
    Ps(:, :, N) = P(:, :, N);
    
    % Backward recursion
    for n = N-1:-1:1
        G = Cp(:, :, n+1)/Pp(:, :, n+1);
        ms(:, n) = m(:, n) + G*(ms(:, n+1) - mp(:, n+1));
        Ps(:, :, n) = P(:, :, n) + G*(Ps(:, :, n+1) - Pp(:, :, n+1))*G';
    end
end
