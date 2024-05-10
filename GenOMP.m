function x = GenOMP(varargin)

% x = GenOMP(D,y,arg);
% x = GenOMP(D,y,arg,N,p);

% N is the max. number of projections to keep in each iteration 
% For N = 1 GenOMP implements OMP. For N > 1 GenOMP implements 
% the Generalized OMP.
% For arg > 1 : arg = nocoeffs
% For arg < 1 : arg = stopping error
% For p > 0 aOMP is implemented along with any other option selected above 

narginchk(3,5);

D = varargin{1};
y = varargin{2};
arg = varargin{3};
if nargin == 5
	N = varargin{4};
	p = varargin{5};
else
	N = 1;
	p = 0;
end

if arg < 0 || arg > size(D,2) || N < 1 || arg > size(D,2) || p < 0 || p > 1
	error('Invalid input arguments');
end
		
err = y;
x = zeros(size(D,2),1);

% aOMP part %
if p > rand(1)
	proj = D'*err;
	[~,mi] = max(abs(proj)./sum(D.^2)');
	D(:,mi) = 0;
end

% Option 1: Stop for a Number of Coefficients or untill the error is negligible %
if arg > 1
	i = 1;
	while (i <= arg/N && norm(err) > eps)
		proj = D'*err;
		[~,mi] = maxk(abs(proj)./sum(D.^2)',N);
		x(mi) = proj(mi);
		x(x ~= 0) = pinv(D(:,(x ~= 0)))*y;
		err = y - D*x;
		i = i + 1;
	end
% Option 2: Stop for a given error %
else
	while norm(err) > arg
		proj = D'*err;
		[~,mi] = maxk(abs(proj)./sum(D.^2)',N);
		x(mi) = proj(mi);
		x(x ~= 0) = pinv(D(:,(x ~= 0)))*y;
		err = y - D*x;
	end
end

end