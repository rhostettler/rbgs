function px = logmvnpdf(x, m, P)
% Logarithm of multivariate normal pdf
%
% Syntax:
%   lpx = LOGMVNPDF(x, m, P)
%
% In:
%   x   N-times-dx vector of values to evaluate
%   m   N-times-dx vector of means (default: 0)
%   P   dx-times-dx covariance matrix (default: I)
% 
% Out:
%   lpx The logarithm of the pdf value
%
% Description:
%   Returns the logarithm of N(x; m, P) (or N(x; 0, I) if m and P are 
%   omitted), that is, the log-likelihood. Everything is calculated in
%   log-domain such that numerical precision is retained to a high degree.
%
%   The arguments x and m are automatically expanded to match each other.
%
%   **Note**: For consistency with Matlab's mvnpdf, each *row* in x (and
%   consequently m) is assumed to be a vector.
%
% See also:
%   mvnpdf

%{
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
    narginchk(1, 3);
    [Nx, dx] = size(x);
    switch nargin
        case 1
            m = zeros([Nx, dx]);
            P = eye(dx);
        case 2
            P = eye(dx);
    end

    %% Expand
    % Some sanity checks
    [Nm, ~] = size(m);
    Nv = size(P, 1);
    
    % Automagically expand the arguments
    if Nx ~= Nm
        if Nx == 1 && Nm > 1
            x = ones(Nm, 1)*x;
        elseif Nx > 1 && Nm == 1
            m = ones(Nx, 1)*m;
        else
            error('Input argument size mismatch');
        end
    end
    
    %% Calculation
    % Cholesky decomposition of the covariance matrix' inverse
    Cinv = P\eye(Nv);
    Linv = chol(Cinv).';
    
    % Efficient calculation of the pdf value
    epsilon = (x-m)*Linv;
    a = -1/2*sum(epsilon.^2, 2);
    b = -1/2*log(det(P));
    c = -Nv/2*log(2*pi);
    px = a + b + c;
end
