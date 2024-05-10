function [Y,im] = im2vect(im,bsize)
% im2vect vectorizes an images using non overlapping square blocks of side length bsize. %
% The size of image must be multiples of bsize to loose no information. %

d1 = size(im,1)/bsize;
d2 = size(im,2)/bsize;
limrow = size(im,1);
if (floor(d1) - d1) ~= 0  
	limrow = size(im,1) - bsize;
end
limcol = size(im,2);
if (floor(d2) - d2) ~= 0
	limcol = size(im,2) - bsize;
end

ind = 1;
for i = 1:bsize:limrow
	for j = 1:bsize:limcol
	block = im(i:i+bsize-1,j:j+bsize-1);
	Y(:,ind) = block(:);
	ind = ind+1;
	end
end
