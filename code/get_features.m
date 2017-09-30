% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function [features] = get_features(image, x, y, feature_width)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature vector should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

%Placeholder that you can delete. Empty features.
% features = zeros(size(x,1), 128);
% 
% filter = fspecial('Gaussian',16,1);
% [Gx,Gy] = imgradientxy(filter,'sobel');
% 
% Ix = imfilter(image,Gx,'symmetric','same','conv');
% Iy = imfilter(image,Gy,'symmetric','same','conv');
% 
% mag = zeros(size(Ix));
% dir = zeros(size(Ix));
% 
% features = [];
% 
% for i = 1:size(Iy,1)
% 	for j = 1:size(Ix,2)
% 		mag(i,j) = sqrt(Ix(i,j)^2 + Iy(i,j)^2);
% 		dir_ini = atand(Iy(i,j)/Ix(i,j));
% 		if (dir_ini <= -67.5)
% 			dir(i,j) = 1;
% 		elseif (dir_ini <= -45)
% 			dir(i,j) = 2;
% 		elseif (dir_ini <= -22.5)
% 			dir(i,j) = 3;
% 		elseif (dir_ini <= 0)
% 			dir(i,j) = 4;
% 		elseif (dir_ini <= 22.5)
% 			dir(i,j) = 5;
% 		elseif (dir_ini <= 45)
% 			dir(i,j) = 6;
% 		elseif (dir_ini <= 67.5)
% 			dir(i,j) = 7;
% 		else
% 			dir(i,j) = 8;
% 		end
% 	end	
% end
% 
% for i = 1:size(x,1)
% 	feature = zeros(1,8);
%     if(y(i,1)-7 <= 0 || x(i,1)-7 <= 0 || y(i,1)+8 > size(mag,1) || x(i,1)+8 > size(mag,2))
%         continue;
%     end
% 	for j = y(i,1)-7:y(i,1)+8
% 		for k = x(i,1)-7:x(i,1)+8
% 			temp = mag(j,k);
% 			temp1 = dir(j,k);
% 			feature(1,temp1) = feature(1,temp1) + temp;
% 		end
% 	end
% 	features = cat(1,features,feature);
% end
% 
% for i = 1:size(features,1)
% 	features(i,:) = features(i,:)/norm(features(i,:),2);
% end

impatch_or = zeros(feature_width,feature_width,size(x,1));
features = zeros(size(x,1), (feature_width/4)^2*8);

filter = fspecial('gaussian',3,0.5);
image_filtered = imfilter (image, filter,'symmetric','same','conv');
[~ , imgrad_orient] = imgradient(image_filtered, 'sobel');

fun = @(block_struct) histcounts(block_struct.data, [-180:45:180]);

for i = 1:size(x,1)   
    impatch_or(:,:,i) = imgrad_orient((y(i)-7):(y(i)+8),(x(i)-7):(x(i)+8));    
    hist_temp = blockproc(impatch_or(:,:,i),[feature_width/4 feature_width/4], fun);
    features(i,:) = reshape(hist_temp',[1 ((feature_width/4)^2)*8]);
    
    features(i,:) = features(i,:)/norm(features(i,:),2);
end
end