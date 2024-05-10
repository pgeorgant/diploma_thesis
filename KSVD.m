% K-SVD (as is)
% Training the dictionary using an online batch of small images. https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz %


%% PARAMETRIZATION
ImageSize = 32;				% number or pixels to select per block  %
Ns = 500;					% number of images in Y %
K = 100;					% number of atoms in D %
J = 100;					% number of algorithm iterations %
iterations = 1;				% TOTAL ITERATIONS to average out for RESULTS %
nocoeffs = 5;				% number of coefficients to keep per block %
rand_update = false;		% randomly update atoms of D when true %

% data_batch_1 contains 10000 32x32 vectorized images (row-wise) %
load('online batches/data_batch_1.mat');

%% TRAINING

keepMSE = zeros(length(Ns),J);
keepTime = zeros(length(Ns),J);
for n = 1:length(Ns)
	N = Ns(n);
	
	Y = double(data(1:N,1:ImageSize)');
	Y = Y - repmat(mean(Y),ImageSize,1);
	Y = normc(Y);

	MSE = zeros(J,N,iterations);
	time = zeros(J,iterations);
	
	for it = 1:iterations
		% D is zero mean Gaussian and column-wise normalized %
		D = randn(ImageSize,K);
		D = D - repmat(mean(D),ImageSize,1);
		D = normc(D);
		D = [ones(ImageSize,1) D(:,2:end)];
		X = zeros(K,N);
		for j = 1:J
			disp(['KSVD, N: ' num2str(Ns(n)) ', try: ' num2str(it) ', iteration: ' num2str(j)]);
			tic1 = tic;
			% ------- K-SVD (as is)-------- %
			parfor i = 1:N
				X(:,i) = GenOMP(D,Y(:,i),nocoeffs);
			end
			for k = pick
				ind = find(X(k,:) ~= 0);
				E = Y - D*X + D(:,k)*X(k,:);
				E_reduced = E(:,ind);
				if ~isempty(ind)
					[u1,s1,v1] = svds(E_reduced,1,'largest');
					D(:,k) = u1;
					X(k,:) = 0;
					X(k,ind) = s1*v1;
				end
			end
			MSE(j,:,it) = sum((Y-D*X).^2)/ImageSize;
			% ----------------------------- %
			time(j+1,it) = time(j,it) + toc(tic1);
		end
	end
	keepMSE(n,:) = mean(mean(MSE,2),3);
	keepTime(n,:) = mean(time(2:end,:),2);
end

save(['RESU\KSVD_' num2str(iterations) 'try_' num2str(J) 'iter_' num2str(nocoeffs) 'coeffs.mat']);