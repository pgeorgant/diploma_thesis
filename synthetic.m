% Conducting the experiment of "Synthtic data". Given a Dictionary (initialized using data from Y) and a random sparse X, we synthesize data Y and try to retrieve D. %

% PARAMETRIZATION %
iterations = 50;		% Total iterations to average out for results %
J = 30;					% Iterations of the training algorithm %
K = 60;					% Number of atoms in the dictinoary %
nocoeffs = 3;			% Number of coefficients %
bsize = 5;				% Block size %
SNRs = [0 10 20 30 50];	% SNR scenarios %
lengthY = 7500;			% Number of vectors in Y %

matchedD_init = zeros(length(SNRs),iterations);
matchedD = zeros(length(SNRs),iterations);
similarity = cell(length(SNRs),iterations);


% Rand data dictionary %
D_init = randn(bsize^2,K);
D_init = D_init - repmat(mean(D_init),bsize^2,1);
D_init = normc(D_init);

for t = 1:length(SNRs)
	for it = 1:iterations
		% Initializing X, Y %
		X = zeros(K,lengthY);
		parfor i = 1:lengthY
			x = [ones(nocoeffs,1); zeros(K-nocoeffs,1)];
			randind = randperm(K);
			X(:,i) = x(randind);
		end
		Y = D_init*X;
		Ps = mean(Y(:).^2);
		Pn = Ps*10^(-SNRs(t)/10);
		Y = Y + Pn*randn(size(Y));
		Y = Y - repmat(mean(Y),bsize^2,1);
		Y = normc(Y);
		
		% Dictionary initialization with synthetic data %
		D = Y(:,1:K);
		
		% Sparse decomposition of Y %
		for j = 1:J
			disp(['iteration ' num2str(j)]);
			parfor i = 1:length(Y)
				X(:,i) = GenOMP(D,Y(:,i),nocoeffs);
			end
			for k = 1:K
				ind = find(X(k,:) ~= 0);
				E = Y - D*X + D(:,k)*X(k,:);
				E_r = E(:,ind);
				if ~isempty(ind)
					[u1,s1,v1] = svds(E_r,1,'largest');
					D(:,k) = u1;
					X(k,:) = 0;
					X(k,ind) = s1*v1;
				end
			end
		end
		similarity{t,it} = abs(D_init'*D);
		err_map = 1 - similarity{t,it};
		errthres = 0.01;
		err_map(err_map > errthres) = 0;
		err_map(err_map > 0) = 1;
		distD_init = sum(err_map,2);
		matchedD_init(t,it) = numel(find(distD_init > 0));
		distD = sum(err_map);
		matchedD(t,it) = numel(find(distD > 0));
	end
end

save('syntheticRand7500.mat');

% Plots %
figure;
load('syntheticRand.mat');
plot(matchedD_init,'.k','LineWidth',0.0001); hold
plot(mean(matchedD_init,2),'--*k','LineWidth',2); hold
xlabel('SNR(dB)','FontSize',12'); ylabel(['#matched/' num2str(K)],'FontSize',12);
xlim([0 length(SNRs)+1]);
ylim([0 K]);
xticks(1:5);
xticklabels({'0','10','20','30','50'});
grid
title('1500 training data points | 50 iter.');
print(gcf,'res\synthetic1500','-dpng','-r300');
figure;
load('syntheticRand7500.mat');
plot(matchedD_init,'.k','LineWidth',0.0001); hold
plot(mean(matchedD_init,2),'-*k','LineWidth',2); hold
xlabel('SNR(dB)','FontSize',12'); ylabel(['#matched/' num2str(K)],'FontSize',12);
xlim([0 length(SNRs)+1]);
ylim([0 K]);
xticks(1:5);
xticklabels({'0','10','20','30','50'});
grid
title('7500 training data points | 50 iter.');

%%
for t = 1:size(similarity,1)
	distD_init = cell(size(similarity,2),1);
	distD = cell(size(similarity,2),1);
	figure;
	tsh = tight_subplot(1,2,[0 0.03],[0.04 0.04],[0.03 0]);
	suptitle(['SNR: ' num2str(SNRs(t)) 'dB | 50 iterations']);
	for it = 1:size(similarity,2)
		err_map = 1 - similarity{t,it};
		errthres = 0.01;
		err_map(err_map > errthres) = 0;
		err_map(err_map > 0) = 1;
		distD_init{it} = sum(err_map,2);
		matchedD_init(t,it) = numel(find(distD_init{it} > 0));
		distD{it} = sum(err_map);
		matchedD(t,it) = numel(find(distD{it} > 0));
	end
	
	p(1:5) = 1-5/50;
	p(6:size(similarity,2)) = 1 - (6:size(similarity,2))/50;
	axes(tsh(1));
	hold on;
	for it = 1:size(similarity,2)
		stem3(1:K,it*ones(1,K),distD_init{it},'filled','Color',p(it)*[1 1 1],'LineStyle','none');
		title(['D_{init}, matched: ' num2str(matchedD_init(t,it)) '/' num2str(K)],'FontSize',12);
		zlabel('Times used','FontSize',12); xlabel('Atoms','FontSize',12); ylabel('Iterations','FontSize',12); zlim([0 K]); zlim([0 2]);
		zticks(0:2);
	end
	grid on;
	view(45, 10);
	hold off;
	axes(tsh(2));
	hold on;
	for it = 1:size(similarity,2)
		stem3(1:K,it*ones(1,K),distD{it},'filled','Color',p(it)*[1 1 1],'LineStyle','none');
		title(['D, matched: ' num2str(matchedD(t,it)) '/' num2str(K)],'FontSize',12);
		zlabel('Times used','FontSize',12); xlabel('Atoms','FontSize',12); ylabel('Iterations','FontSize',12); zlim([0 K]); zlim([0 2]);
		zticks(0:2);
	end
	grid on;
	view(45, 10);hold off;
	set(gcf,'Position',[1 31 1920 973]);
	print(gcf,['res\syntheticMatchesSNR' num2str(SNRs(t))],'-dpng','-r300');
end