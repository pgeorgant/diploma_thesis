% Comparison of OMP aOMP and GenOMP time comsumption as a function of sparsity %

%% Loading the images and training the dictionary using K-SVD %
dicttrain

Y = Y(:,sum(abs(Y))~=0);

%% Sparse Decomposition run time collection
N = length(Y);
K = 100;
nocoeffs = 2:2:12;
err1 = zeros(length(nocoeffs),1);
err2 = zeros(length(nocoeffs),1);
err3 = zeros(length(nocoeffs),1);
omptime1 = zeros(length(nocoeffs),1);
omptime2 = zeros(length(nocoeffs),1);
omptime3 = zeros(length(nocoeffs),1);

for t = 1:length(nocoeffs)
	disp(['nocoeffs: ' num2str(nocoeffs(t))])
	noc = nocoeffs(t);
	% OMP
	X1 = zeros(K,N);
	omptic1 = tic;
	for i = 1:N
		X1(:,i) = GenOMP(D,Y(:,i),noc);
	end
	omptime1(t) = toc(omptic1);
	err1(t) = mean(mean((Y - D*X1).^2));
	% aOMP
	X2 = zeros(K,N);
	omptic2 = tic;
	for i = 1:N
		X2(:,i) = GenOMP(D,Y(:,i),noc,1,0.5);
	end
	omptime2(t) = toc(omptic2);
	err2(t) = mean(mean((Y - D*X2).^2));
	% gOMP
	X3 = zeros(K,N);
	omptic3 = tic;
	for i = 1:N
		X3(:,i) = GenOMP(D,Y(:,i),noc,2,0);
	end
	omptime3(t) = toc(omptic3);
	err3(t) = mean(mean((Y - D*X3).^2));
end
err = [err1 err2 err3];
omptime = [omptime1 omptime2 omptime3];

%% Plots

figure;
h = plot(nocoeffs,10*log10(err));
title(['Mean MSE (dB) over ' num2str(length(Y)) ' blocks']);
set(h,{'Color'},{'k';'k';'k'},{'LineStyle'},{'-';'--';'-.'},'LineWidth',1);
legend({'OMP','aOMP','gOMP'},'FontSize',12);
xlabel(['Sparsity (/' num2str(size(Y,1)) ')']);
ylabel('Mean MSE (dB)');
xticks(nocoeffs);
xlim([1,13]);

figure;
h = plot(nocoeffs,omptime);
title(['Time consumption (' num2str(length(Y)) ' blocks)']);
set(h,{'Color'},{'k';'k';'k'},{'LineStyle'},{'-';'--';'-.'},'LineWidth',1);
legend({'OMP','aOMP','gOMP'},'FontSize',12,'Location','NW');
xlabel(['Sparsity (/' num2str(size(Y,1)) ')']);
ylabel('Time (sec.)');
xticks(nocoeffs);
xlim([1,13]);