function alpha = catrand(lw, N)
% Sampling from categorical random variable
%
% Syntax:
%   alpha = CATRAND(lw, N)
%
% In:
%   lw      Log-weights of the categorical random variable
%   N       Number of samples to draw (default: 1)
%
% Out:
%   alpha   Random samples
%
% Description:
%   Sample N samples from a categorical distribution of M categories
%   defined by the log-weights lw.
%
%   **Note:** This is a limited categorical sampling routine which requires
%   that N <= M. The method is mainly intended for resampling in sequential
%   Monte Carlo methods.

%{
% Copyright (C) 2016-present Roland Hostettler
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
    narginchk(1, 2);
    if nargin < 2 || isempty(N)
        N = 1;
    end

    %% Sample
    % Calculate weights
    w = exp(lw-max(lw));
    w = w/sum(w, 2);
    
    % Sample
    M = size(w, 2);
    alpha = zeros(1, M);
    k = 0;
    u = 1/M*rand(1);
    for j = 1:M
        % Get the no. of times we need to replicate this sample
        L = floor(M*(w(j)-u)) + 1;
        
        % Replicate the sample
        alpha(k+1:k+L) = j*ones(1, L);
        k = k + L;
        
        % Update threshold
        u = u + L/M - w(j);
    end
    
    % Randomize sample and select a subset, if applicable
    alpha = alpha(randperm(M, N));
end
