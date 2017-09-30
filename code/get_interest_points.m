% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width, scale)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% Placeholder that you can delete -- random points
% x = ceil(rand(500,1) * size(image,2));
% y = ceil(rand(500,1) * size(image,1));

filter = fspecial('Gaussian',3,0.5);
[Gx,Gy] = imgradientxy(filter,'sobel');

Ix = imfilter(image,Gx,'symmetric','same','conv');
Iy = imfilter(image,Gy,'symmetric','same','conv');

Ix2 = Ix.*Ix;
Iy2 = Iy.*Iy;
Ixy = Ix.*Iy;

filter = fspecial('Gaussian',3,1);
GIx2 = imfilter(Ix2,filter,'symmetric','same','conv');
GIy2 = imfilter(Iy2,filter,'symmetric','same','conv');
GIxy = imfilter(Ixy,filter,'symmetric','same','conv');

alpha = 0.05;
Har = (GIx2.*GIy2 - GIxy.*GIxy) - alpha*((GIx2 + GIy2).*(GIx2 + GIy2));

threshold = 1.54e-3*scale;
Har_th = Har>threshold;
Har_th1 = Har.*Har_th;

Har_supp = imregionalmax(Har_th1);
Har_supp1 = Har.*Har_supp;

Har_th_supp = (Har_th1==Har_supp1).*Har_supp1;
[y,x,confidence] = find(Har_th_supp>0);

low_ind = feature_width/2 - 1;
high_ind = feature_width/2;

ind_rmx = find ((x-low_ind) < 1);
x(ind_rmx) = [];
y(ind_rmx) = [];
ind_rmy = find((y-low_ind) < 1);
x(ind_rmy) = [];
y(ind_rmy) = [];
ind_rmx = find ((x+high_ind) > size(image,2));
x(ind_rmx) = [];
y(ind_rmx) = [];
ind_rmy = find ((y+high_ind) > size(image,1));
x(ind_rmy) = [];
y(ind_rmy) = [];

figure, imagesc(image), axis image, colormap(gray), hold on
plot(x,y,'ys'), title('corners detected');

end