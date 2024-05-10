% Conducting the experiment of reconstructing noisy images using an already trained dictionary. %

% tight_subplot function being used can be found here: https://www.mathworks.com/matlabcentral/fileexchange/27991-tight_subplot-nh-nw-gap-marg_h-marg_w %

% Training the dictionary %
%----------%
dicttrain 
% -- OR -- %
load('Dcats_wDC');
%----------%

K = 100;			% Number of atoms in the dictionary %
K = 100;			% Number of atoms in the dictionary %
nocoeffs = 5;		% Numver of coefficients per block %
bsize = 8;			% Block size %

%% Reconstruction testing

files = dir('newcats\test\*.jpg');
SNR = [-10 0 10];					% SNR scenarios (in dB)%
mse = cell(length(SNR),1);
R = zeros(1,length(files));
na = cell(length(files),1);

for k = 1:length(files)
	im = rgb2gray(im2double(imread(['newcats\test\' files(k).name])));
	imwrite(im,'tmpres\im.jpg');
	imp = double(imread('tmpres\im.jpg'));
	figure(k);
	tsh{k} = tight_subplot(2,2,[0 0],[0 0],[0 0]);
	axes(tsh{k}(1));
	imagesc(imp); colormap('gray');
	xticks(''); yticks('');
	na{k} = strtok(files(k).name,'.');
	for n = 1:length(SNR)
		[Y_test,~] = im2vect(im,bsize);
		Y_test = Y_test + sqrt(var(Y_test(:))/(10^(SNR(n)/10)))*randn(size(Y_test));
		disp(['SNR:' num2str(SNR(n)) 'dB, Image: ' files(k).name]);
		
		% Sparse coding %
		X_test = zeros(K,size(Y_test,2));
		parfor i = 1:size(Y_test,2)
			X_test(:,i) = GenOMP(D,Y_test(:,i),nocoeffs);
		end
		rec_Y = D*X_test;
		
		rec_im = vect2im(rec_Y,0,size(im));

		imwrite(rec_im,'tmpres\rec_im.jpg');
		rec_im = double(imread('tmpres\rec_im.jpg'));
		
		mse{n}(k) = mean((im(:)/255 - rec_im(:)/255).^2);
		
		figure(k);

		axes(tsh{k}(n+1));
		imagesc(rec_im); colormap('gray');
		xticks(''); yticks('');
		drawnow
	end
	set(gcf, 'Position', [20 229 1173 737]);
end

figure;
colormap([0 0 0;0.3 0.3 0.3;0.6 0.6 0.6]);
hb = barh(cell2mat(mse)','grouped');
hb(3).FaceColor = [0 0 0];
hb(2).FaceColor = [0.3 0.3 0.3];
hb(1).FaceColor = [0.6 0.6 0.6];
lg = legend('-10dB','0dB','10dB');
lg.FontSize = 12;
yticklabels(na);
xlabel('MSE','FontSize',16);
set(gca,'TickLabelInterpreter','none');
set(gcf, 'Position', [20 229 1173 737]);
