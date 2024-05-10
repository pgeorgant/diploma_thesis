% Dictionary Training on real images of cats using K-SVD %

% Parameters %
noblocks = 500; % Number of blocks to take per image %
bsize = 8;		% Block size %
K = 100;		% Number of atoms in the dictionary %
J = 30;			% Training algorthm iterations%
nocoeffs = 5;	% Number of non zero coefficients to keep per block %

% Loading and Vectorizing images %
files = dir('newcats\train\*.jpg');
Y = zeros(bsize^2,length(files)*noblocks);
tsh = tight_subplot(3,5,[0 0],[0 0],[0 0]);
for i = 1:length(files)
	im = rgb2gray(im2double(imread(['newcats\train\' files(i).name])));
	axes(tsh(i));imagesc(im); colormap('gray'); xticks(''); yticks('');

	[Y_all,~] = im2vect(im,bsize);
	ind = randperm(size(Y_all,2));
	Y(:,(i-1)*noblocks+1:i*noblocks) = Y_all(:,ind(1:noblocks));

end

% Training the dictionary %

X = zeros(K,length(files)*noblocks);
D = randn(bsize^2,K);
D = D - repmat(mean(D),bsize^2,1);
D = normc(D);
D = [ones(bsize^2,1) D(:,2:end)];
figure(1);
imagesc(vect2im(D,0,[80,80])); colormap('gray'); xticks(''); yticks('');
figure(2);
imagesc(D'*D > 0.9); colormap('gray'); xticks(''); yticks('');
drawnow;

for j = 1:J
	disp(['iteration ' num2str(j)]);
	parfor i = 1:length(files)*noblocks
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
	figure(1);
	imagesc(vect2im(D,0,[80,80])); colormap('gray'); xticks(''); yticks('');
	figure(2);
	imagesc(D'*D > 0.9); colormap('gray'); xticks(''); yticks('');
	drawnow;
end

save('D','Dcats_wDC');

% Plots %

figure;
imagesc(vect2im(D,0,[80 80])); colormap('gray');
title('D: Trained w/ 15 images x 5000 8x8 blocks');
xticks(''); yticks('');
print(gcf,'res\Dcats8','-dpng','-r300');
figure;
for i = 1:length(D)
	imagesc(reshape(D(:,i),8,8)); title(num2str(i)); colormap('gray');
	drawnow
	pause
end

figure;
A = D'*D;
[h,c] = imhist(A);
bar(c,h/numel(A),'FaceColor','k');
xlabel('Correlation','FontSize',14); ylabel('pdf','FontSize',14);
title('Autocorrelation pdf of the trained dictionary','FontSize',14);

figure;
imagesc(abs(D'*D)); colormap('gray');
title('Dictionary Autocorrelation Matrix');
xlabel('Atoms of D'); ylabel('Atoms of D');
xticks('');yticks('');

figure;
imagesc(abs(D'*D)>0.8); colormap('gray');
title('Dictionary Autocorrelation Matrix (> 0.8)');
xlabel('Atoms of D'); ylabel('Atoms of D');
xticks('');yticks('');

figure;
subplot(2,2,[1,2]);
mse = mean((Y-D*X).^2);
[max_mse,max_ind] = max(mse);
stem(Y(:,max_ind),'filled','Color',[0.6 0.6 0.6]); hold;
stem(D*X(:,max_ind),'filled','k','LineWidth',2); hold;
legend('Original block','Reconstructed');
xlim([0 bsize^2+1]);
title(['Block with Max. Error (~=' num2str(max_mse) ')']);
xlabel('Pixel'); ylabel('Intensity');
subplot(2,2,[3,4]);
hb = bar(X(:,max_ind),'histc');
set(hb,'FaceColor','k');
title('Corresponding sparse vector');
xlabel('Atom'); ylabel('Weight')
xlim([0 K]);

figure;
subplot(2,2,[1,2]);
[min_mse,min_ind] = min(mse(mse>0));
stem(Y(:,min_ind),'filled','Color',[0.6 0.6 0.6]); hold;
stem(D*X(:,min_ind),'filled','k','LineWidth',2); hold;
legend('Original block','Reconstructed');
xlim([0 bsize^2+1]);
title(['Block with Min. Error (~=' num2str(min_mse) ')']);
xlabel('Pixel'); ylabel('Intensity');
subplot(2,2,[3,4]);
hb = bar(X(:,min_ind),'histc');
set(hb,'FaceColor','k');
title('Corresponding sparse vector');
xlabel('Atom'); ylabel('Weight')
xlim([0 K]);
