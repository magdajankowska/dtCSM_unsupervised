function [P, beta] = x2p(X, u, tol)
%X2P Identifies appropriate sigma's to get kk NNs up to some tolerance 
%
%   [P, beta] = x2p(xx, kk, tol)
% 
% Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
% kernel with a certain uncertainty for every datapoint. The desired
% uncertainty can be specified through the perplexity u (default = 15). The
% desired perplexity is obtained up to some tolerance that can be specified
% by tol (default = 1e-4).
% The function returns the final Gaussian kernel in P, as well as the 
% employed precisions per instance in beta.
% %IMPORTANT: after adding scalling of data, for each instance separately, beta values
%             cannot be compared between each other, and so are rather meaningless
%M. Jankowska: added scalling of data - see below

    
    if ~exist('u', 'var') || isempty(u)
        u = 15;
    end
    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-4; 
    end
    
    % Initialize some variables
    n = size(X, 1);                     % number of instances
    P = zeros(n, n);                    % empty probability matrix
    beta = ones(n, 1);                  % empty precision vector
    logU = log(u);                      % log of perplexity (= entropy)
    
    % Compute pairwise distances
    disp('Computing pairwise distances...');
    sum_X = sum(X .^ 2, 2);
    D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * X * X'));

    % Run over all datapoints
    disp('Computing P-values...');
    for i=1:n
        
        %if ~rem(i, 500)
        %    disp(['Computed P-values ' num2str(i) ' of ' num2str(n) ' datapoints...']);
        %end
        
        % Set minimum and maximum values for precision
        betamin = -Inf; 
        betamax = Inf;
        
        % Compute the Gaussian kernel and entropy for the current precision
        Di = D(i, [1:i-1 i+1:end]) ;
        %disp(['for instance: ' num2str(i)]);
		%disp(["min(min(Di)) orig " num2str(min(min(Di)))]);
		%disp(["max(max(Di)) orig " num2str(max(max(Di)))]);

		%M.Jankowska:
		%if the distances (Di) are large, the values of 
		%the exponent funciton of the Gausian kernel are exactly zero,
		%due to the numerical precision
		%If that happens for all Di, i.e., even if the minimum of Di is too
		%large to produce non-zero value of the exponent function,
		%then the normalization factor in Hbeta function become 0,
		%and so through division by zero the P-values become NaN 
		%In order to fix it, the distances are scalled by dividing them
		%by the minimum of Di (unless this mininum is less than 1.):
		
		scale=max(1,min(max(Di,0)));
        Di = Di ./ scale;
        
        %Notice that if distances were small enough, so the "division by zero"
        %would not appear even without scalling, then scalling does not affect P values obtained below
        %(they are not necessary identical to the ones that would be obtained without scalling, 
        %due to the numerical precision and stopping condition, 
        %but are obtained in exactly the same procedure)
        %but it does affect beta values (see code below)
        %As such, with separate scalling of each instance, beta values cannot be compared between instances.
        
		%disp(['scale ' num2str(scale)]);
		%disp(["min(min(Di)) scaled " num2str(min(min(Di)))]);
		%disp(["max(max(Di)) scaled " num2str(max(max(Di)))]);

        [H, thisP] = Hbeta(Di, beta(i));
        
        % Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while abs(Hdiff) > tol && tries < 50
            
            % If not, increase or decrease precision
            if Hdiff > 0
                betamin = beta(i);
                if isinf(betamax)
                    beta(i) = beta(i) * 2;
                else
                    beta(i) = (beta(i) + betamax) / 2;
                end
            else
                betamax = beta(i);
                if isinf(betamin) 
                    beta(i) = beta(i) / 2;
                else
                    beta(i) = (beta(i) + betamin) / 2;
                end
            end
            
            % Recompute the values
            [H, thisP] = Hbeta(Di, beta(i));
            Hdiff = H - logU;
            tries = tries + 1;
        end
        
        % Set the final row of P
        P(i, [1:i - 1, i + 1:end]) = thisP;
    end    
    
    %Notice that beta values (and so sigma values) cannot be now compared
    %between instances, as they were obtained for data scalled
    %independently for each instance
    %disp(['Mean value of sigma: ' num2str(mean(sqrt(1 ./ beta)))]);
    %disp(['Minimum value of sigma: ' num2str(min(sqrt(1 ./ beta)))]);
    %disp(['Maximum value of sigma: ' num2str(max(sqrt(1 ./ beta)))]);
end
    


% Function that computes the Gaussian kernel values given a vector of
% squared Euclidean distances, and the precision of the Gaussian kernel.
% The function also computes the perplexity of the distribution.
function [H, P] = Hbeta(D, beta)
    P = exp(-D * beta);
    sumP = sum(P);
    H = log(sumP) + beta * sum(D .* P) / sumP;
    P = P / sumP;
end

