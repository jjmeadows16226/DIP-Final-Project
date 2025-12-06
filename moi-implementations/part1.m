clear; 
clc; 
close all;

I = imread('/Users/moisesgomez/Developer/DIP_FinalProject/DIP-Final-Project/IMG_4951.png');
hsvI = rgb2hsv(I);

H = hsvI(:,:,1);
S = hsvI(:,:,2);
V = hsvI(:,:,3);

% yellowish_pixels = (H >= 0.01 & H <= 0.15) & (S >= 0.1 & S <= 1) & (V >= 0.2 & V <= 1);
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
% edgeThreshold = bwmorph(edgeThreshold, "thin", inf);

% Remove edges from original binary mask
severed = (1-edgeThreshold) & disks;

%% Remove "yellowish" pixels (i.e., branches and flower petals) from mask
% disks = imgaussfilt(double(disks), 3) > 0.20;
% disks = imopen(disks, se);
% disks = imgaussfilt(double(disks), 2) > 0.20;
% disks = edges>350 | disks;
yelloish_pixels = imdilate(yellowish_pixels, se);

% disks_filtered = disks;
% disks_filtered(yellowish_pixels) = 0;

disks_filtered = disks & (1-yellowish_pixels);
disks_filtered = disks_filtered & (1-edgeThreshold);

disks_filtered(shadows) = 0;

% disks_filtered = disks_filtered & (1-(H > 0.1 & H < 0.25));
% disks_filtered = disks_filtered & (1-(edges>50));

roundness = bwdist(Gmag > 100) > 10;
disks_filtered(roundness) = 0;
%% Label binary regions and filter
[L, numRegions] = bwlabel(disks_filtered);
all_areas = accumarray(L(L > 0), 1, [numRegions 1]);
min_size = 80;

stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area', 'Perimeter');

ecc   = [stats.Eccentricity];
ca = [stats.ConvexArea];
ar = [stats.Area];
peri = [stats.Perimeter];

orca = ca ./ ar;

% Filter based on area, eccentricity, and occupy rate convex area
% & ecc' <= 0.99 & orca' <= 3.3
 % & orca' <= 3.8
valid_indices = find(all_areas > min_size & ecc' <= 0.98)';
filteredRegions = ismember(L, valid_indices);

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

% [L, numRegions] = bwlabel(out);

% stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area');
% all_areas = accumarray(L(L > 0), 1, [numRegions 1]);
% 
% valid_indices = find(all_areas < 7900)';
% filteredRegions = ismember(L, valid_indices);

filteredRegions = out;

pbinary(filteredRegions, "filteredRegions - out")
lko = filteredRegions;
filledMask = imfill(lko, 'holes');


% greenish_pixels(filteredRegions) = 0;

%% Find circles using circular Hough Transform
% Find circles (sensitity of 0.9 too low, >= 0.94 works well)
% filteredRegions = imgaussfilt(double(filteredRegions), 3) > 0.20;
% "Sensitivity", 0.97, [20 50]
[centers, radii] = imfindcircles(filledMask, [30 50], "Sensitivity", 0.94);
% size(radii)
% Overlay detected circles
figure('Name', 'Initial Circles')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii,1)))
hold on
viscircles(centers, radii, 'LineWidth', 3);
hold off

centers_clean = [];
radii_clean = [];

debug_image = zeros(h,w);
debug_image2 = zeros(h,w);


% uuu = V < 0.3 & (1-disks_filtered);
% greenish_pixels = greenish_pixels | uuu;

% greenish_pixels(imdilate(filledMask, strel('disk', 3))) = 0;
greenish_pixels = 1-filledMask;

for k = 1:numel(radii)
    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k) .* 0.4);

    rowRange = max(1, cy - r) : min(h, cy  + r);
    colRange = max(1, cx - r) : min(w, cx + r);
    patch = V(rowRange, colRange);
    
    patch_region = filledMask(rowRange, colRange);
    mask_ratio = nnz(patch_region) / numel(patch);

    patch_region_green = greenish_pixels(rowRange, colRange);
    greenish_ratio = nnz(patch_region_green) / numel(patch);

    yyy = greenChannel-blueChannel;
    petal_mask = yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92;

    just_petals_mask = yyy(rowRange, colRange) > 92 & (1-patch_region);
    just_petals = nnz(just_petals_mask);

    just_disk_mask = yyy(rowRange, colRange) < 20;
    just_disks = nnz(just_disk_mask);

    % if mask_ratio > 0.3 && greenish_ratio < 0.95
    % if (greenish_ratio ./ mask_ratio) < 2
    if greenish_ratio <= 0.3
        centers_clean = [centers_clean; centers(k,:)];
        radii_clean = [radii_clean; radii(k)];
    end

    debug_image(rowRange, colRange) = greenish_ratio + 0.001;
    debug_image2(rowRange, colRange) = mask_ratio + 0.001;
end

figure('Name', 'Initial Circles Clean')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii_clean,1)))
hold on
viscircles(centers_clean, radii_clean, 'LineWidth', 3);
hold off

[CCC, RRR, GGG] = dip_findfiltercircles(filledMask, [30 50], 0.94, 0.4, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 1, 0, 1, 0, 0.3]);

% size(RRR, 1)
% size(radii_clean,1)

%% Remove, refind

% output mask
mask = false(h, w);

% coordinate grid
% [xg, yg] = meshgrid(1:w, 1:h);
% 
for k = 1:size(centers,1)
    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k));

    rowRange = max(1, cy - r) : min(h, cy + r);
    colRange = max(1, cx - r) : min(w, cx + r);
 % ./ 0.8
    % bw_patch = filteredRegions(rowRange, colRange);
    % patch_dist = bwdist(bw_patch);

    % r = round(r + mean(patch_dist(patch_dist>0)));

    % rowRange = max(1, cy - r) : min(h, cy + r);
    % colRange = max(1, cx - r) : min(w, cx + r);

    mask(rowRange, colRange) = true;



    % add this circle to the mask
    % mask = mask | ones(rowRange, colRange);
end

% mask = dip_createcirclemask(centers, radii, 1, [h,w]);

filteredRegions = filteredRegions & (1-mask);

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




% [L, numRegions] = bwlabel(filteredRegions);
% all_areas = accumarray(L(L > 0), 1, [numRegions 1]);
% min_size = 35;
% 
% stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area', 'Perimeter');
% 
% ecc   = [stats.Eccentricity];
% ca = [stats.ConvexArea];
% ar = [stats.Area];
% peri = [stats.Perimeter];
% 
% orca = ca ./ ar;
% 
% valid_indices = find(all_areas < round(pi*((30+13)*(30+13))))';
% filteredRegions = ismember(L, valid_indices);


[centers, radii] = imfindcircles(filteredRegions, [20 45], "Sensitivity", 0.92);

% Overlay detected circles
figure('Name', 'Initial Circles 2')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii,1)))
hold on
viscircles(centers, radii, 'LineWidth', 3);
hold off



centers_clean2 = [];
radii_clean2 = [];

debug_image3 = zeros(h,w);
debug_image4 = zeros(h,w);
debug_image5 = zeros(h,w);
debug_image6 = zeros(h,w);
debug_image7 = zeros(h,w);

circle_props = [];
k_count = 1;
k_centerOfMass = [];


% [centers_clean2, radii_clean2] = dip_findfiltercircles(filteredRegions, [20 45], 0.92, 0.4, V, filteredRegions, greenish_pixels, G-B, [20, 92, 92, 20], [0, 0.8]);
[centers_clean2, radii_clean2, k_centerOfMass] = dip_findfiltercircles(filteredRegions, [20 45], 0.92, 0.4, V, filteredRegions, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 0.8]);

%{
for k = 1:numel(radii)
    myMask = false(h,w);

    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k) .* 0.4);

    rowRange = max(1, cy - (r)) : min(h, cy + (r));
    colRange = max(1, cx - (r)) : min(w, cx + (r));
    patch = V(rowRange, colRange);
    
    patch_region = filteredRegions(rowRange, colRange);
    mask_ratio = nnz(patch_region) / numel(patch);

    myMask(rowRange, colRange) = patch_region;

    patch_region_green = greenish_pixels(rowRange, colRange);
    greenish_ratio = nnz(patch_region_green) / numel(patch);

    yyy = greenChannel-blueChannel;
    petal_mask = yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92;
    fraction_2 = nnz(petal_mask) / numel(patch);

    just_petals_mask = yyy(rowRange, colRange) > 92 & (1-patch_region);
    just_petals = nnz(just_petals_mask);

    just_disk_mask = yyy(rowRange, colRange) < 20;
    just_disks = nnz(just_disk_mask);

    patch_dir = Gmag(rowRange, colRange);

    if nnz(myMask) == 0 
        continue
    end

    stats_m = regionprops(myMask, 'Centroid');
    regionCentroid = stats_m.Centroid;   % [x, y]

    if (just_petals / numel(patch)) < 0.8
        centers_clean2 = [centers_clean2; centers(k,:)];
        radii_clean2 = [radii_clean2; radii(k)];

        k_centerOfMass = [k_centerOfMass; regionCentroid];
    % elseif greenish_ratio / (mask_ratio + 1e-6) > 26 && greenish_ratio / (mask_ratio + 1e-6) < 50
    % elseif (just_petals / numel(patch)) > 0.4
    %     centers_clean2 = [centers_clean2; centers(k,:)];
    %     radii_clean2 = [radii_clean2; radii(k)];
    % 
    %     k_centerOfMass = [k_centerOfMass; regionCentroid];
    end


    % centers_band = 

    % r2  = round(r / 0.8);
    % rowRange2 = max(1, cy - (r2)) : min(h, cy + (r2));
    % colRange2 = max(1, cx - (r2)) : min(w, cx + (r2));
    % myMask(rowRange, colRange) = myMask & (1-)

    debug_image3(rowRange, colRange) = (just_petals / numel(patch)) + 0.001;
    debug_image4(rowRange, colRange) = (just_disks / numel(patch)) + 0.001;
    debug_image5(rowRange, colRange) = greenish_ratio + 0.001;
    debug_image6(rowRange, colRange) = mask_ratio + 0.001;
    debug_image7(rowRange, colRange) = sum(patch_dir(:));
end
%}

figure('Name', 'Initial Circles Clean 2')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii_clean2,1)))
hold on
viscircles(centers_clean2, radii_clean2, 'LineWidth', 3);

plot(centers_clean2(:,1), centers_clean2(:,2), 'gx', 'MarkerSize', 6, 'LineWidth', 1.5);  % green crosses
plot(k_centerOfMass(:,1), k_centerOfMass(:,2), 'bs', 'MarkerSize', 6, 'LineWidth', 1.5);  % blue squares

hold off


[centers_merged, radii_merged] = dip_mergecircles(centers_clean2, radii_clean2, k_centerOfMass, 0.6, 25);

%{
N = size(centers_clean2, 1);

% Build an adjacency matrix of overlaps
overlapMatrix = false(N, N);

reduction = 0.6;
radii_clean2 = radii_clean2 .* reduction;


for i = 1:N
    for j = i+1:N
        d = norm(centers_clean2(i,:) - centers_clean2(j,:));
        if d < (radii_clean2(i) + radii_clean2(j))
            overlapMatrix(i,j) = true;
            overlapMatrix(j,i) = true;
        end
    end
end

% Find connected components (groups of mutually overlapping circles)
G_matrix = graph(overlapMatrix);
groups = conncomp(G_matrix);

% Figure out which circles to delete
toDelete = false(N,1);

% radii_clean2 = radii_clean2 ./ reduction;

for g = unique(groups)
    idx = find(groups == g);
    
    %{
    % Only consider groups with actual overlaps
    if numel(idx) > 1
        % [~, largestIndex] = max(radii_clean2(idx));
        % toDelete(idx(largestIndex)) = true;
        [~, largestRatio] = max(circle_props(idx));

               % Mark all *other* circles in the group for deletion
        % idx(largestIndex) = [];     % remove the largest
        toDelete(idx(largestRatio)) = true;       % delete the smaller ones
    end
    %}

    % Skip groups without overlaps
    if numel(idx) < 2
        continue
        % winner = idx;
    end

    % Step 1: find the maximum property value in this group
    % groupProps = circle_props(idx);
    % maxProp = max(groupProps);



    % Step 2: find all circles that have this maximum property
    % candidates = idx(groupProps == maxProp);

    % Step 3: tie-breaker if more than one circle has the same max property
    % Here we just pick the first, but you could use radius, index, etc.
    % winner = candidates(1);
    % var(groupProps)
    % if var(groupProps) < 0.03

    % if maxProp > 0.6 && var(groupProps) < 0.03
    %     [~, idxBiggest] = max(radii_clean2(idx));
    %     winner = idx(idxBiggest);
    % else
        % winner = idx(1);

        diffs = centers_clean2(idx,:) - k_centerOfMass(idx, :);     % Nx2 matrix
        dists = sqrt(sum(diffs.^2, 2));              % Nx1 distances

        [minDist, ix] = min(dists);
        % winner = idx(ix);

        if minDist < 25
            winner = idx(ix);
        else
            winner = idx(1);
            % disp('hi')size(
            % [~, idxBiggest] = max(radii_clean2(idx));
            % winner = idx(idxBiggest);
        end
    % end

    % 406
    % if ismember(249, idx)
    %     candidates
    %     winner
    %     idx
    %     idxBiggest
    %     % disp('hi')
    %     maxProp
    %     mSTD = std(groupProps)
    %     mVar = var(groupProps)
    %     var(groupProps) / numel(groupProps)
    % end

    % Step 4: delete all other circles in the group
    toDelete(setdiff(idx, winner)) = true;
end

radii_clean2 = radii_clean2 ./ reduction;

% Remove them
centers_merged = centers_clean2(~toDelete, :);
radii_merged   = radii_clean2(~toDelete);

%}

figure('Name', 'Merged Circles (Clean 2)')
imshow(rgbImage, [])
title(sprintf('Merged Circles: %d', size(radii_merged,1)))
hold on
viscircles(centers_merged, radii_merged, 'LineWidth', 3);
hold off



for k = 1:size(centers_merged,1)
    cx = round(centers_merged(k,1));
    cy = round(centers_merged(k,2));
    r  = round(radii_merged(k));

    rowRange = max(1, cy - r) : min(h, cy + r);
    colRange = max(1, cx - r) : min(w, cx + r);
 % ./ 0.8
    % bw_patch = filteredRegions(rowRange, colRange);
    % patch_dist = bwdist(bw_patch);

    % r = round(r + mean(patch_dist(patch_dist>0)));

    % rowRange = max(1, cy - r) : min(h, cy + r);
    % colRange = max(1, cx - r) : min(w, cx + r);

    mask(rowRange, colRange) = true;



    % add this circle to the mask
    % mask = mask | ones(rowRange, colRange);
end

filteredRegions = filteredRegions & (1-mask);

[centers, radii] = imfindcircles(filteredRegions, [25 50], "Sensitivity", 0.96);

% Overlay detected circles
figure('Name', 'Initial Circles 3')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii,1)))
hold on
viscircles(centers, radii, 'LineWidth', 3);
hold off


centers_clean3 = [];
radii_clean3 = [];

debug_image8 = zeros(h,w);
debug_image9 = zeros(h,w);
debug_image10 = zeros(h,w);
debug_image11 = zeros(h,w);
debug_image12 = zeros(h,w);

for k = 1:numel(radii)
    myMask = false(h,w);

    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k));

    rowRange = max(1, cy - (r)) : min(h, cy + (r));
    colRange = max(1, cx - (r)) : min(w, cx + (r));
    patch = V(rowRange, colRange);
    
    patch_region = filteredRegions(rowRange, colRange);
    mask_ratio = nnz(patch_region) / round(pi*(r*r));

    myMask(rowRange, colRange) = patch_region;

    patch_region_green = greenish_pixels(rowRange, colRange);
    greenish_ratio = nnz(patch_region_green) / round(pi*(r*r));

    yyy = greenChannel-blueChannel;
    petal_mask = yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92;
    fraction_2 = nnz(petal_mask) / numel(patch);

    just_petals_mask = yyy(rowRange, colRange) > 92 & (1-patch_region);
    just_petals = nnz(just_petals_mask);

    just_disk_mask = yyy(rowRange, colRange) < 20;
    just_disks = nnz(just_disk_mask);

    patch_dir = Gdir(rowRange, colRange);


    stats_m = regionprops(myMask, 'Centroid');
    regionCentroid = stats_m.Centroid;   % [x, y]

    if mask_ratio > 0.4 && (just_disks / numel(patch)) > 0.3
        centers_clean3 = [centers_clean3; centers(k,:)];
        radii_clean3 = [radii_clean3; radii(k)];

        k_centerOfMass = [k_centerOfMass; regionCentroid];
    end

    debug_image8(rowRange, colRange) = (just_petals / numel(patch)) + 0.001;
    debug_image9(rowRange, colRange) = (just_disks / numel(patch)) + 0.001;
    debug_image10(rowRange, colRange) = greenish_ratio + 0.001;
    debug_image11(rowRange, colRange) = mask_ratio + 0.001;
    debug_image12(rowRange, colRange) = sum(patch_dir(:));
end


% Overlay detected circles
figure('Name', 'Initial Circles 3 CLEAN')
imshow(rgbImage, [])
title(sprintf('Initial Circles: %d', size(radii_clean3,1)))
hold on
viscircles(centers_clean3, radii_clean3, 'LineWidth', 3);
hold off

% 
% % output mask
% mask2 = false(h, w);
% 
% % coordinate grid
% [xg, yg] = meshgrid(1:w, 1:h);
% 
% for k = 1:size(centers,1)
%     cx = centers(k,1);
%     cy = centers(k,2);
%     r  = radii(k);
% 
%     % add this circle to the mask
%     mask2 = mask2 | ((xg - cx).^2 + (yg - cy).^2 <= r^2);
% end
% 
% filteredRegions = filteredRegions & (1-mask2);
% 
% [centers, radii] = imfindcircles(filteredRegions, [20 50], "Sensitivity", 0.80);
% 
% % Overlay detected circles
% figure('Name', 'Initial Circles 3')
% imshow(rgbImage, [])
% title(sprintf('Initial Circles: %d', size(radii,1)))
% hold on
% viscircles(centers, radii, 'LineWidth', 3);
% hold off


%% Rasterize each circle and generate binary mask for each
% Precompute a coordinate grid
[xGrid, yGrid] = meshgrid(1:w, 1:h);

% Loop over circles
masks = {};  % 3-D logical array

% for k = 1:numel(radii)
%     cx = centers(k,1);
%     cy = centers(k,2);
%     r  = radii(k);
% 
%     % Distance from each pixel to the circle center
%     % Very time consuming
%     dist = (xGrid - cx).^2 + (yGrid - cy).^2;
% 
%     % Create mask for this circle
%     masks{k} = dist <= r^2;
% end

%% Overlay mask in red over orginal image
[~, finalNum] = bwlabel(filteredRegions);
mask = logical(filteredRegions);
R(mask) = 0;
G(mask) = 0;
B(mask) = 255;

overlay = cat(3, R, G, B);

figure
imshow(overlay, [])
title(finalNum)

%% NEXT: Use color, region props, etc., to further filter disks

%{
% centers: Nx2 matrix
% radii:   Nx1 vector
% overlap_threshold: positive number requiring extra overlap

%{
overlap_threshold = 50;

N = size(centers,1);
A = zeros(N);

for i = 1:N
    for j = i+1:N
        d = norm(centers(i,:) - centers(j,:));
        if d < (radii(i) + radii(j) - overlap_threshold)
            A(i,j) = 1;
            A(j,i) = 1;
        end
    end
end

G = graph(A);
bins = conncomp(G);   % gives cluster labels

unique_bins = unique(bins);
merged_centers = [];
merged_radii = [];

for b = unique_bins
    idx = find(bins == b);

    % --- Case 1: Cluster has only one circle → keep it ---
    if isscalar(idx)
        merged_centers = [merged_centers; centers(idx,:)];
        merged_radii   = [merged_radii; radii(idx)];
        continue
    end

    % --- Case 2: Cluster has multiple circles → merge them ---
    pts = [];
    for k = idx
        theta = linspace(0,2*pi,200)';    % better sampling
        pts = [pts; centers(k,:) + radii(k)*[cos(theta), sin(theta)]];
    end

    [C,R] = minboundcircle(pts(:,1), pts(:,2));
    merged_centers = [merged_centers; C];
    merged_radii   = [merged_radii; R];
end

% Overlay detected circles
figure
imshow(I, [])
hold on
viscircles(merged_centers, merged_radii, 'LineWidth', 3);
hold off

%}

% ++++++++++++++++++++
interm_centers = [];
interm_radii = [];


circle_props = [];
k_count = 1;
k_centerOfMass = [];

for k = 1:numel(radii)
    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k));

    rowRange = max(1, cy - r) : min(h, cy + r);
    colRange = max(1, cx - r) : min(w, cx + r);
    patch = V(rowRange, colRange);
    
    patch_region = filteredRegions(rowRange, colRange);

    patch_region_green = greenish_pixels(rowRange, colRange);

    stats = regionprops(patch_region, 'Centroid');
    regionCentroid = stats.Centroid;   % [x, y]

    mask_ratio = sum(patch_region(:)) / (pi*(r*r));

    greenish_ratio = nnz(patch_region_green) / (pi*(r*r));

    min_patch = min(patch(:));
    
    [vv1, vv2] = size(patch);

    patch_mask = patch < 0.12 & patch > 0.01;
    fraction = nnz(patch_mask) / (vv1*vv2);

    yyy = greenChannel-blueChannel;
    petal_mask = yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92;
    fraction_2 = nnz(petal_mask) / (vv1*vv2);

    just_petals_mask = yyy(rowRange, colRange) > 92 & (1-patch_region);
    just_petals = nnz(just_petals_mask);

    just_disk_mask = yyy(rowRange, colRange) < 20;
    just_disks = nnz(just_disk_mask);

   num_petals = nnz(petal_mask);
   num_non_petals = numel(patch) - num_petals;


    % reg_disk(rowRange, colRange) = k;
    reg_disk_2(rowRange, colRange) = k;

    cond = true;
    if (pi*(r*r)) > 1521 && nnz(patch_region) > 650
        % if just_disks / just_petals < 0.7
        if just_petals / nnz(patch) > 0.3 && just_petals / nnz(patch) < 0.52
            cond = false;
        end
        
    end

    if mask_ratio > 0.035 && (fraction_2 >= 0.0932)
        
        rowOuter = max(1, min(rowRange)-2) : min(h, max(rowRange)+2);
        colOuter = max(1, min(colRange)-2) : min(w, max(colRange)+2);

        top    = [rowOuter(1)      * ones(1,numel(colOuter)); colOuter];
        bottom = [rowOuter(end)    * ones(1,numel(colOuter)); colOuter];
        left   = [rowOuter(:)';     colOuter(1) * ones(1,numel(rowOuter))];
        right  = [rowOuter(:)';     colOuter(end) * ones(1,numel(rowOuter))];

        borderIdx = unique([top, bottom, left, right]', 'rows');

        rrr = borderIdx(:,1);
        ccc = borderIdx(:,2);
        
        lin = sub2ind(size(I), rrr, ccc);

        chnl = greenChannel-blueChannel;
        borderPixels = chnl(lin);
        borderPixels_v = V(lin);

        patch_edges = edges(lin);

        if mean(borderPixels) > 30
            interm_centers = [interm_centers; centers(k,:)];
            interm_radii = [interm_radii; radii(k)];

            circle_props = [circle_props; mask_ratio];
            k_centerOfMass = [k_centerOfMass; regionCentroid];
            k_count = k_count + 1;
        end
    else
    end
end

% ++++++++++++++++++++

% ====================

centers = interm_centers;
radii = interm_radii;

N = size(centers, 1);

% Build an adjacency matrix of overlaps
overlapMatrix = false(N, N);

reduction = 0.4;
radii = radii .* reduction;


for i = 1:N
    for j = i+1:N
        d = norm(centers(i,:) - centers(j,:));
        if d < (radii(i) + radii(j))
            overlapMatrix(i,j) = true;
            overlapMatrix(j,i) = true;
        end
    end
end

% Find connected components (groups of mutually overlapping circles)
G = graph(overlapMatrix);
groups = conncomp(G);

% Figure out which circles to delete
toDelete = false(N,1);

radii = radii ./ reduction;

for g = unique(groups)
    idx = find(groups == g);
    
    %{
    % Only consider groups with actual overlaps
    if numel(idx) > 1
        % [~, largestIndex] = max(radii(idx));
        % toDelete(idx(largestIndex)) = true;
        [~, largestRatio] = max(circle_props(idx));

               % Mark all *other* circles in the group for deletion
        % idx(largestIndex) = [];     % remove the largest
        toDelete(idx(largestRatio)) = true;       % delete the smaller ones
    end
    %}

    % Skip groups without overlaps
    if numel(idx) < 2
        continue
        % winner = idx;
    end

    % Step 1: find the maximum property value in this group
    groupProps = circle_props(idx);
    maxProp = max(groupProps);



    % Step 2: find all circles that have this maximum property
    candidates = idx(groupProps == maxProp);

    % Step 3: tie-breaker if more than one circle has the same max property
    % Here we just pick the first, but you could use radius, index, etc.
    % winner = candidates(1);
    % var(groupProps)
    % if var(groupProps) < 0.03

    % if maxProp > 0.6 && var(groupProps) < 0.03
    %     [~, idxBiggest] = max(radii(idx));
    %     winner = idx(idxBiggest);
    % else
        % winner = idx(1);

        diffs = centers(idx,:) - k_centerOfMass(idx, :);     % Nx2 matrix
        dists = sqrt(sum(diffs.^2, 2));              % Nx1 distances

        [minDist, ix] = min(dists);
        % winner = idx(ix);

        if minDist < 2
            winner = idx(ix);
        else
            winner = idx(1);
            % [~, idxBiggest] = max(radii(idx));
            % winner = idx(idxBiggest);
        end
    % end

    % 406
    % if ismember(249, idx)
    %     candidates
    %     winner
    %     idx
    %     idxBiggest
    %     % disp('hi')
    %     maxProp
    %     mSTD = std(groupProps)
    %     mVar = var(groupProps)
    %     var(groupProps) / numel(groupProps)
    % end

    % Step 4: delete all other circles in the group
    toDelete(setdiff(idx, winner)) = true;
end

% Remove them
centers_clean = centers(~toDelete, :);
radii_clean   = radii(~toDelete);

% ====================

centers = centers_clean;
radii = radii_clean;

endImage = zeros(h, w);
endImage_2 = zeros(h, w);

reg_disk = zeros(h, w);
reg_disk_2 = zeros(h, w);
reg_peta = zeros(h, w);
reg_peta_2 = zeros(h, w);

final_centers = [];
final_radii = [];

interm_centers = [];
interm_radii = [];

circle_props = [];

k_count = 1;
k_centerOfMass = [];

for k = 1:numel(radii)
    cx = round(centers(k,1));
    cy = round(centers(k,2));
    r  = round(radii(k));

    rowRange = max(1, cy - r) : min(h, cy + r);
    colRange = max(1, cx - r) : min(w, cx + r);
    patch = V(rowRange, colRange);
    
    patch_region = filteredRegions(rowRange, colRange);

    patch_region_green = greenish_pixels(rowRange, colRange);

    stats = regionprops(patch_region, 'Centroid');
    regionCentroid = stats.Centroid;   % [x, y]

    mask_ratio = sum(patch_region(:)) / (pi*(r*r));

    greenish_ratio = nnz(patch_region_green) / (pi*(r*r));

    if greenish_ratio > 0.6
        % disp('hi')
        continue
    end

    min_patch = min(patch(:));
    
    [vv1, vv2] = size(patch);

    patch_mask = patch < 0.12 & patch > 0.01;
    fraction = nnz(patch_mask) / (vv1*vv2);

    yyy = greenChannel-blueChannel;
    petal_mask = yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92;
    fraction_2 = nnz(petal_mask) / (vv1*vv2);

    just_petals_mask = yyy(rowRange, colRange) > 92 & (1-patch_region);
    just_petals = nnz(just_petals_mask);

    just_disk_mask = yyy(rowRange, colRange) < 20;
    just_disks = nnz(just_disk_mask);

   num_petals = nnz(petal_mask);
   num_non_petals = numel(patch) - num_petals;


    % reg_disk(rowRange, colRange) = k;
    reg_disk_2(rowRange, colRange) = k;

    % if  mask_ratio > 0.1 && num_non_petals > num_petals && num_non_petals/(num_petals + 0.01) > 10 && num_non_petals/(num_petals + 0.01) < 11 && ~(num_non_petals > num_petals && (fraction >= 0.0968) && mask_ratio > 0.1)
    %     pbinary(patch, [fraction, mask_ratio])
    %     pbinary(yyy(rowRange, colRange) > 50, [num_non_petals, num_petals])
    %     pbinary(patch_mask)
    % end

    cond = true;
    if (pi*(r*r)) > 1521 && nnz(patch_region) > 650
        % if just_disks / just_petals < 0.7
        if just_petals / nnz(patch) > 0.3
            cond = false;
        end
        
    end

    % if k == 136
    %     fraction
    %     mask_ratio
    %     fraction_2
    %     cond
    % 
    %     sum(patch_region(:))
    %     pbinary(just_petals_mask)
    %     pbinary(patch)
    %     pbinary(patch_region)
    % 
    %     pause
    % end
    % 0.045

    if (fraction >= 0.033) && mask_ratio > 0.1 && (fraction_2 >= 0.0932)
        
        rowOuter = max(1, min(rowRange)-2) : min(h, max(rowRange)+2);
        colOuter = max(1, min(colRange)-2) : min(w, max(colRange)+2);

        top    = [rowOuter(1)      * ones(1,numel(colOuter)); colOuter];
        bottom = [rowOuter(end)    * ones(1,numel(colOuter)); colOuter];
        left   = [rowOuter(:)';     colOuter(1) * ones(1,numel(rowOuter))];
        right  = [rowOuter(:)';     colOuter(end) * ones(1,numel(rowOuter))];

        borderIdx = unique([top, bottom, left, right]', 'rows');

        rrr = borderIdx(:,1);
        ccc = borderIdx(:,2);
        
        lin = sub2ind(size(I), rrr, ccc);

        chnl = greenChannel-blueChannel;
        borderPixels = chnl(lin);
        borderPixels_v = V(lin);

        patch_edges = edges(lin);

        % petals_mask = chnl > 100;
        % petals_mask = petals_mask(lin);
        % 
        % fraction = nnz(petals_mask) / size(lin,1);
        % 


        % borderPixels = borderPixels(borderPixels > 0);

        reg_peta(lin) = mean(borderPixels);
        reg_peta_2(lin) = nnz(borderPixels>50) / numel(lin);


        if (mean(borderPixels) > 40 || median(borderPixels) == 0) && mean(patch_edges) > 60
            
            endImage(rowRange, colRange) = redChannel(rowRange, colRange);

            final_centers = [final_centers; centers(k,:)];
            final_radii   = [final_radii; radii(k)];
            circle_props = [circle_props; mask_ratio];
            k_centerOfMass = [k_centerOfMass; regionCentroid];


            reg_disk(rowRange, colRange) = k_count;
            k_count = k_count + 1;
        else
            endImage_2(lin) = redChannel(lin);

            interm_centers = [interm_centers; centers(k,:)];
            interm_radii   = [interm_radii; radii(k)];

            % pbinary(chnl(rowOuter, colOuter), [mean(borderPixels), mean(patch_edges), nnz(borderPixels>50) / numel(lin)])
        end
    else
        % 
        % pbinary(yyy(rowRange, colRange), [fraction, mask_ratio])
        % pbinary(yyy(rowRange, colRange) < 20 | yyy(rowRange, colRange) > 92, [num_non_petals, num_petals])
        % pbinary(patch_mask)
    end
    % imshow(patch,[])

    % Create mask for this circle
    % masks{k} = patch;
end

% Overlay detected circles
figure('Name', 'Intermediate Circles')
imshow(rgbImage, [])
title(sprintf('Intermediate Circles: %d', size(interm_radii,1)))
hold on
viscircles(interm_centers, interm_radii, 'LineWidth', 3);
hold off



% Overlay detected circles
figure('Name', 'Final Circles')
imshow(rgbImage, [])
title(sprintf('Final Circles: %d', size(final_radii,1)))
hold on
viscircles(final_centers, final_radii, 'LineWidth', 3);
hold off


% centers: Nx2 array of [x, y]
% radii:   Nx1 vector

% final_centers = centers;
% final_radii = radii;
% 
% circle_props = resize(circle_props, numel(final_radii));
% k_centerOfMass = resize(k_centerOfMass, numel(final_radii));

%{
N = size(final_centers, 1);

% Build an adjacency matrix of overlaps
overlapMatrix = false(N, N);

reduction = 0.4;
final_radii = final_radii .* reduction;


for i = 1:N
    for j = i+1:N
        d = norm(final_centers(i,:) - final_centers(j,:));
        if d < (final_radii(i) + final_radii(j))
            overlapMatrix(i,j) = true;
            overlapMatrix(j,i) = true;
        end
    end
end

% Find connected components (groups of mutually overlapping circles)
G = graph(overlapMatrix);
groups = conncomp(G);

% Figure out which circles to delete
toDelete = false(N,1);

final_radii = final_radii ./ reduction;

for g = unique(groups)
    idx = find(groups == g);
    
    %{
    % Only consider groups with actual overlaps
    if numel(idx) > 1
        % [~, largestIndex] = max(final_radii(idx));
        % toDelete(idx(largestIndex)) = true;
        [~, largestRatio] = max(circle_props(idx));

               % Mark all *other* circles in the group for deletion
        % idx(largestIndex) = [];     % remove the largest
        toDelete(idx(largestRatio)) = true;       % delete the smaller ones
    end
    %}

    % Skip groups without overlaps
    if numel(idx) < 2
        continue
        % winner = idx;
    end

    % Step 1: find the maximum property value in this group
    groupProps = circle_props(idx);
    maxProp = max(groupProps);



    % Step 2: find all circles that have this maximum property
    candidates = idx(groupProps == maxProp);

    % Step 3: tie-breaker if more than one circle has the same max property
    % Here we just pick the first, but you could use radius, index, etc.
    % winner = candidates(1);
    % var(groupProps)
    % if var(groupProps) < 0.03

    % if maxProp > 0.6 && var(groupProps) < 0.03
    %     [~, idxBiggest] = max(final_radii(idx));
    %     winner = idx(idxBiggest);
    % else
        % winner = idx(1);

        diffs = final_centers(idx,:) - k_centerOfMass(idx, :);     % Nx2 matrix
        dists = sqrt(sum(diffs.^2, 2));              % Nx1 distances

        [minDist, ix] = min(dists);
        % winner = idx(ix);

        if minDist < 2
            winner = idx(ix);
        else
            winner = idx(1);
            % [~, idxBiggest] = max(final_radii(idx));
            % winner = idx(idxBiggest);
        end
    % end

    % 406
    % if ismember(249, idx)
    %     candidates
    %     winner
    %     idx
    %     idxBiggest
    %     % disp('hi')
    %     maxProp
    %     mSTD = std(groupProps)
    %     mVar = var(groupProps)
    %     var(groupProps) / numel(groupProps)
    % end

    % Step 4: delete all other circles in the group
    toDelete(setdiff(idx, winner)) = true;
end

% Remove them
centers_clean = final_centers(~toDelete, :);
radii_clean   = final_radii(~toDelete);
%}


% Overlay detected circles
figure('Name', 'Merged Circles')
imshow(rgbImage, [])
title(sprintf('Merged Circles: %d', size(radii_clean,1)))
hold on
viscircles(centers_clean, radii_clean, 'LineWidth', 3);
hold off
%}