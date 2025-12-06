clear; 
clc; 
close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');
% I = imread('/Users/moisesgomez/Downloads/IMG_8154.jpeg');

tic
hsvI = rgb2hsv(I);

H = hsvI(:,:,1);
S = hsvI(:,:,2);
V = hsvI(:,:,3);

yellowish_pixels = (H >= 0 & H <= 0.15) & (V >= 0.17 & V <= 1);
greenish_pixels = H >= 0.19 & H <= 0.45;

R = I(:,:, 1);
G = I(:,:, 2);
B = I(:,:, 3);

rgbImage = I;
I = rgb2gray(I);
[h, w] = size(I);

I = double(I);

redChannel   = R;
greenChannel = G;
blueChannel  = B;


%% Use color channels to produce rough binary mask
greenMinusBlue = G-B;
greenMinusBlue = greenMinusBlue < 30;

shadows = (G-B) > 10 & (G-B) < 30;
shadows = imgaussfilt(double(shadows), 3) > 0.95;

greenMinusRed = G-R;
greenMinusRed = greenMinusRed > 0;

disks = greenMinusBlue & (1-greenMinusRed);

% disks(shadows) = 0;

%% Sever shadow regions from disks using edges
edges = edge_8(G-B, 0.05);
[Gmag, Gdir] = imgradient(G-B);
edgeThreshold = Gmag > 20;

% Thicken thresholded edges
se = strel('disk', 1);
edgeThreshold = imerode(edgeThreshold, se);

% Remove edges from original binary mask
severed = (1-edgeThreshold) & disks;

%% Remove "yellowish" pixels (i.e., branches and flower petals) from mask
yellowish_pixels = imdilate(yellowish_pixels, se);

disks_filtered = disks & (1-yellowish_pixels);
disks_filtered = disks_filtered & (1-edgeThreshold);

disks_filtered(shadows) = 0;

% roundness = bwdist(Gmag > 100) > 10;
% disks_filtered(roundness) = 0;

disks_filtered = imgaussfilt(double(disks_filtered), 3) > 0.3;

%% Label binary regions and filter
[L, numRegions] = bwlabel(disks_filtered);
all_areas = accumarray(L(L > 0), 1, [numRegions 1]);

stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area');

min_size = 80;
ecc   = [stats.Eccentricity];
ca = [stats.ConvexArea];
ar = [stats.Area];
orca = ca ./ ar;

% Filter based on area, eccentricity, and occupy rate convex area\
valid_indices = find(all_areas > min_size & ecc' <= 0.98)';
filteredRegions = ismember(L, valid_indices);

% For certain regions, replace with its convex hull
[L, numRegions] = bwlabel(filteredRegions);
out = false(size(filteredRegions));
for k = 1:numRegions
    % Isolate this component
    regionMask = (L == k);
    if sum(regionMask(:)) <= 3000 && sum(regionMask(:)) > 60
        % Compute its convex hull (only for this component)
        hull = bwconvhull(regionMask);
        % Insert the hull into the output
        out = out | hull;
    else
         out = out | regionMask;
    end
end

filteredRegions = out;

filtered_copy = filteredRegions;
filledMask = imfill(filtered_copy, 'holes');

 greenish_pixels = 1-filledMask;

%% BEGIN

[centers_i1, radii_i1, centroids_i1] = dip_findfiltercircles(filledMask, [30 50], 0.95, 0.4, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 1, 0, 1, 0, 0.6, 0, 1, -1, 0.1]);
[centers_i1_merged, radii_i1_merged, centroids_i1_merged] = dip_mergecircles(centers_i1, radii_i1, centroids_i1, 0.6, 25);


% Supress found circles in original image
mask = dip_createcirclemask(centers_i1_merged, radii_i1_merged, 1.1, [h,w]);
filledMask = filledMask & ~mask;

%{
fR_copy = filteredRegions;
[L, numRegions] = bwlabel(filteredRegions);

out = false(size(filteredRegions));
for k = 1:numRegions
    % Isolate this component
    regionMask = (L == k);
    if sum(regionMask(:)) <= 3000 && sum(regionMask(:)) > 60
        % Compute its convex hull (only for this component)
        hull = bwconvhull(regionMask);
        % Insert the hull into the output
        out = out | hull;
    else
         out = out | regionMask;
    end
end

filteredRegions = out;
%}

pbinary(filledMask)

% [centers_i2_initial, radii_i2_initial] = imfindcircles(filledMask, [20 45], 'Sensitivity', 0.93);                                                                    
[centers_i2, radii_i2, centroids_i2] = dip_findfiltercircles(filledMask, [20 45], 0.93, 1, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 0.5, 0, 1, 0, 1, 0, 1, 0.02, 1, -1, 1]);
[centers_i2_merged, radii_i2_merged, centroids_i2_merged] = dip_mergecircles(centers_i2, radii_i2, centroids_i2, 0.6, 25);

% Supress found circles in original image
mask = dip_createcirclemask(centers_i2_merged, radii_i2_merged, 1.1, [h,w]);
filledMask = filledMask & ~mask;

[centers_i3, radii_i3, centroids_i3] = dip_findfiltercircles(filledMask, [20 35], 0.94, 1, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 0.5, 0, 1, 0, 1, 0, 1, 0.02, 1, -1, 1]);
toc

%% Overlay found circles on image

% Iteration 1
figure('Name', 'Iteration 1')
imshow(rgbImage, [])
title(sprintf('Number of Circles: %d', size(radii_i1,1)))
hold on
viscircles(centers_i1, radii_i1, 'LineWidth', 3);
hold off

% Iteration 2
figure('Name', 'Iteration 2')
imshow(rgbImage, [])
title(sprintf('Number of Circles: %d', size(radii_i2,1)))
hold on
viscircles(centers_i2, radii_i2, 'LineWidth', 3);
hold off

% Iteration 3
figure('Name', 'Iteration 3')
imshow(rgbImage, [])
title(sprintf('Number of Circles: %d', size(radii_i3,1)))
hold on
viscircles(centers_i3, radii_i3, 'LineWidth', 3);
hold off

centers_total = [centers_i1_merged; centers_i2_merged; centers_i3];
radii_total   = [radii_i1_merged; radii_i2_merged; radii_i3];
centroids_total = [centroids_i1_merged; centroids_i2_merged; centroids_i3];

[centers_total_merged, radii_total_merged, ~] = dip_mergecircles(centers_total, radii_total, centroids_total, 0.6, 25);

% Total Circles
figure('Name', 'TOTAL')
imshow(rgbImage, [])
title(sprintf('Number of Circles: %d', size(radii_total_merged,1)))
hold on
viscircles(centers_total_merged, radii_total_merged, 'LineWidth', 3, 'Color', [0 0 1]);
hold off


%% Overlay mask in red over orginal image
[~, finalNum] = bwlabel(filtered_copy);
mask = logical(filtered_copy);
R(mask) = 0;
G(mask) = 0;
B(mask) = 255;

overlay = cat(3, R, G, B);

figure
imshow(overlay, [])
title(finalNum)


[a, b] = part1(rgbImage, [0 0.15], [0.17 1], [20 50], 80, 0.98, 1000, [60 3000], [0.95 0.93 0.94]);

% Total Circles Standalone Function
figure('Name', 'TOTAL Standalone')
imshow(rgbImage, [])
title(sprintf('Number of Circles: %d', size(b,1)))
hold on
viscircles(a, b, 'LineWidth', 3, 'Color', [0 1 0]);
hold off