% LS-COPER (An AppK-SVD derivation)
% Training the dictionary using an online batch of small images. https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz %

clear
close all
clc

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
keepMSE2 = zeros(length(Ns),J);

keepTime2 = zeros(length(Ns),J);

for n = 1:length(Ns)
	N = Ns(n);
	
	Y = double(data(1:N,1:ImageSize)');
	Y = Y - repmat(mean(Y),ImageSize,1);
	Y = normc(Y);

	MSE2 = zeros(J,N,iterations);
	time2 = zeros(J,iterations);
	
	for it = 1:iterations
	
		% D is zero mean Gaussian and column-wise normalized with an additional DC block at the begining%
		D2 = randn(ImageSize,K-1);
		D2 = D2 - repmat(mean(D2),ImageSize,1);
		D2 = normc(D2);
		D2 = [ones(ImageSize,1) D2];
		X2 = zeros(K,N);
		for j = 1:J
			disp(['COPER, N: ' num2str(Ns(n)) ', try: ' num2str(it) ', iteration: ' num2str(j)]);
			tic3 = tic;
 			% ------------ LS-COPER ------------ %
			for i = 1:N
				X2(:,i) = GenOMP(D2,Y(:,i),nocoeffs);
			end
			for k = pick
 				E2 = Y - D2*X2 + D2(:,k)*X2(k,:);
 				if norm(X2(k,:)) ~= 0
 					D2(:,k) = (X2(k,:)*E2')/norm(X2(k,:))^2;
 				end
 				nd = norm(D2(:,k));
 				D2(:,k) = D2(:,k)/nd;
 				X2(k,:) = X2(k,:)*nd;
 			end
 			MSE2(j,:,it) = sum((Y-D2*X2).^2)/ImageSize;
 			% ---------------------------------- %
			time2(j+1,it) = time2(j,it) + toc(tic3);
		end
	end
 	keepMSE2(n,:) = mean(mean(MSE2,2),3);
	keepTime2(n,:) = mean(time2(2:end,:),2);
end

save(['RESU\COPER_' num2str(iterations) 'try_' num2str(J) 'iter_' num2str(nocoeffs) 'coeffs.mat']);