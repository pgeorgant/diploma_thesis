% Conducting the "Impainting" experiment of filling in missing pixels using an already trained dictionary %

% tight_subplot function being used can be found here: https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w %

% Training the dictionary %
%----------%
dicttrain 
% -- OR -- %
load('Dcats_wDC');
%----------%

% PARAMETRIZATION %
K = 100;			% Number of atoms in the dictionary %
nocoeffs = 5;		% Numver of coefficients per block %
r = [0.2 0.6 0.8];  % Percentage of missing pixels %
bsize = 8;			% Block size %

files = dir('newcats\test\*.jpg');
MSE = cell(length(r),1);
na = cell(length(files),1);

for k = 1:length(files)
	% Loading test image %
	im = rgb2gray(im2double(imread(['newcats\test\' files(k).name])));
	[Y_test_init,~] = im2vect(im,bsize);
	figure(k);
	tsh{k} = tight_subplot(2,2,[0 0],[0 0],[0 0]);
	axes(tsh{k}(1));
	imagesc(im); colormap('gray');
	xticks(''); yticks('');
	for rc = 1:length(r)

		% Distrorting test image %
		randind = randperm(bsize^2);
		randind = randind(1:floor(r(rc)*bsize^2));
		invind = setdiff(1:bsize^2,randind);
		Y_dist = Y_test_init;
		Y_dist(randind,:) = 300;
		Y_test = Y_test_init(invind,:);
		im_dist = vect2im(Y_dist,0,size(im));
		im_dist(im_dist > 255) = 0;

		% Recovering test image %
		Di = D(invind,:);
		parfor i = 1:size(Y_test,2)
			X(:,i) = GenOMP(Di,Y_test(:,i),nocoeffs);
		end
		Y_rec = D*X;
		rec_im = vect2im(Y_rec,0,size(im));

		imwrite(im,'tmpres\im.jpg');
		im = double(imread('tmpres\im.jpg'));
		imwrite(rec_im,'tmpres\rec_im.jpg');
		rec_im = double(imread('tmpres\rec_im.jpg'));
		
		% Calculating Mean Squared Reconstruction Error %
		MSE{rc}(k) = mean((im(:)/255 - rec_im(:)/255).^2);

		axes(tsh{k}(rc+1));
		imagesc(rec_im); colormap('gray');
		xticks(''); yticks('');
		drawnow
	end
	set(gcf, 'Position', [246 277 1175 689]);
	na{k} = strtok(files(k).name,'.');
	
end

figure;
colormap([0 0 0;0.3 0.3 0.3;0.6 0.6 0.6]);
hb = barh(10*log10(cell2mat(MSE)'),'grouped');
hb(3).FaceColor = [0 0 0];
hb(2).FaceColor = [0.3 0.3 0.3];
hb(1).FaceColor = [0.6 0.6 0.6];
lg = legend('20%','60%','80%');
lg.FontSize = 14;
yticklabels(na);
xlabel('MSE(dB)','FontSize',14);
set(gca,'TickLabelInterpreter','none');
set(gcf, 'Position', [20 229 1173 737]);