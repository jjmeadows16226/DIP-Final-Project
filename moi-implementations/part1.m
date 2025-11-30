clear; 
clc; 
close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');
hsvI = rgb2hsv(I);

H = hsvI(:,:,1);
S = hsvI(:,:,2);
V = hsvI(:,:,3);

yellowish_pixels = (H >= 0.01 & H <= 0.15) & (S >= 0.1 & S <= 1) & (V >= 0.2 & V <= 1);

R = I(:,:, 1);
G = I(:,:, 2);
B = I(:,:, 3);

I = rgb2gray(I);
[h, w] = size(I);

I = double(I);

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
% se = strel('disk', 2);
% edgeThreshold = imerode(edgeThreshold, se);

% Remove edges from original binary mask
severed = (1-edgeThreshold) & disks;

%% Remove "yellowish" pixels (i.e., branches and flower petals) from mask
disks_filtered = disks & (1-yellowish_pixels);

%% Label binary regions and filter
[L, numRegions] = bwlabel(disks_filtered);
all_areas = accumarray(L(L > 0), 1, [numRegions 1]);

min_size = 10;

stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area');

ecc   = [stats.Eccentricity];
ca = [stats.ConvexArea];
ar = [stats.Area];

orca = ca ./ ar;

% Filter based on area, eccentricity, and occupy rate convex area
valid_indices = find(all_areas > min_size & ecc' <= 0.98 & orca' <= 4.15)';
filteredRegions = ismember(L, valid_indices);

%% Find circles using circular Hough Transform
% Find circles (sensitity of 0.9 too low, >= 0.94 works well)
[centers, radii] = imfindcircles(filteredRegions, [17 50], "Sensitivity", 0.94);

% Overlay detected circles
figure
imshow(I, [])
hold on
viscircles(centers, radii, 'LineWidth', 3);
hold off

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
    % Very time consuming
    dist = (xGrid - cx).^2 + (yGrid - cy).^2;

    % Create mask for this circle
    masks{k} = dist <= r^2;
end

%% Overlay mask in red over orginal image
mask = logical(filteredRegions);
R(mask) = 255;
G(mask) = 0;
B(mask) = 0;

overlay = cat(3, R, G, B);

figure
imshow(overlay, [])

%% NEXT: Use color, region props, etc., to further filter disks