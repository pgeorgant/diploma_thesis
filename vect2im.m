function [rec_im] = vect2im(rec_Y,meanY,imsize)

% vect2im converts a matrix supposedly containing vectorized blocks of an image back to the original image size. %
% If there is a meanY vector of the mean value of each block to add in while reshaping the data then meanY should be that, otherwise zero. %

bsize = floor(sqrt(size(rec_Y,1)));
rec_Y = rec_Y + repmat(meanY,bsize^2,1);
ind = 1;
for i = 1:bsize:imsize(1)
	for j = 1:bsize:imsize(2)
		rec_im(i:i+bsize-1,j:j+bsize-1) = reshape(rec_Y(:,ind),bsize,bsize);
		ind = ind+1;
	end
end