clear; clc; close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');

R = I(:,:, 1);
G = I(:,:, 2);
B = I(:,:, 3);

I = rgb2gray(I);
[h, w] = size(I);

%% Use color channels to produce rough binary mask
greenMinusBlue = G-B;
greenMinusBlue = greenMinusBlue < 30;

greenMinusRed = G-R;
greenMinusRed = greenMinusRed > 0;

disks = greenMinusBlue & (1-greenMinusRed);

%% Sever shadow regions from disks using edges
edges = edge_8(G-B, 0.05);
edgeThreshold = edges > 40;

% Thicken thresholded edges
se = strel('disk', 2);
edgeThreshold = imdilate(edgeThreshold, se);

% Remove edges from original binary mask
severed = (1-edgeThreshold) & disks;

%% Label binary regions and filter
[L, numRegions] = bwlabel(severed);
all_areas = accumarray(L(L > 0), 1, [numRegions 1]);

min_size = 1000;
max_size = (h*w) / 3;

stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area');

ecc   = [stats.Eccentricity];
ca = [stats.ConvexArea];
ar = [stats.Area];

orca = ca ./ ar;

valid_indices = find( ...
    all_areas > min_size & ...
    all_areas < max_size)';

% for idx = valid_indices
%     submask = (L == idx);
%     imshow(submask, [])
% end

finalDisks = ismember(L, valid_indices);

%% Find circles using circular Hough Transform
% Find circles
[centers, radii] = imfindcircles(finalDisks, [20 45], "Sensitivity", 0.96);

% Overlay detected circles
imshow(finalDisks)
hold on
viscircles(centers, radii, 'LineWidth', 3);

%% Rasterize each circle and generate binary mask for each
% Precompute a coordinate grid
[xGrid, yGrid] = meshgrid(1:w, 1:h);

% Loop over circles
masks = {};  % 3-D logical array

for k = 1:numel(radii)
    cx = centers(k,1);
    cy = centers(k,2);
    r  = radii(k);

    % Distance from each pixel to the circle center
    dist = (xGrid - cx).^2 + (yGrid - cy).^2;

    % Create mask for this circle
    masks{k} = dist <= r^2;
end

%% NEXT: Use color, region props, etc., to further filter disks