% AppK-SVD
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
keepMSE1 = zeros(length(Ns),J);

keepTime1 = zeros(length(Ns),J);

for n = 1:length(Ns)
	N = Ns(n);
	
	Y = double(data(1:N,1:ImageSize)');
	meanY = mean(Y);
	Y = Y - repmat(meanY,ImageSize,1);
	Y = normc(Y);

	MSE1 = zeros(J,N,iterations);
	time1 = zeros(J,iterations);
	
	for it = 1:iterations
		% D is zero mean Gaussian and column-wise normalized with a DC block at the begining %
		D1 = randn(ImageSize,K);
		D1 = D1 - repmat(mean(D1),ImageSize,1);
		D1 = normc(D1);
		D1 = [ones(ImageSize,1) D1(:,2:end)];
		X1 = zeros(K,N);
		for j = 1:J
			disp(['AppKSVD, N: ' num2str(Ns(n)) ', try: ' num2str(it) ', iteration: ' num2str(j)]);
			tic2 = tic;
			% ------- approximate K-SVD ------- %
			for i = 1:N
				X1(:,i) = GenOMP(D1,Y(:,i),nocoeffs);
			end
			for k = pick
				ind = find(X1(k,:) ~= 0);
				Ek = Y - D1*X1 + D1(:,k)*X1(k,:);
				E1 = Ek(:,ind);
				if norm(X1(k,:)) ~= 0
					D1(:,k) = (E1*X1(k,ind)')/norm(E1*X1(k,ind)');
				end
				X1(k,ind) = E1'*D1(:,k);
			end
			MSE1(j,:,it) = sum((Y-D1*X1).^2)/ImageSize;
			% ---------------------------------- %
			time1(j+1,it) = time1(j,it) + toc(tic2);
		end
	end
	keepMSE1(n,:) = mean(mean(MSE1,2),3);
	keepTime1(n,:) = mean(time1(2:end,:),2);
end

save(['AppKSVD_' num2str(iterations) 'try_' num2str(J) 'iter_' num2str(nocoeffs) 'coeffs.mat']);