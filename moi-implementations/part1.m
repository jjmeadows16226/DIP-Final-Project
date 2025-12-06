function [centers, radii] = part1(I, Hrange, Vrange, flowerRadius, minROISize, maxROIEccentricity, maxROIORCA, sizeConvexHull, sensitivities)
    %% Initialize variables and extract color channels
    hsvImage = rgb2hsv(I);
    rgbImage = I;

    H = hsvImage(:,:,1);
    S = hsvImage(:,:,2);
    V = hsvImage(:,:,3);

    R = rgbImage(:,:, 1);
    G = rgbImage(:,:, 2);
    B = rgbImage(:,:, 3);

    greenChannel = G;
    blueChannel  = B;

    yellowish_pixels = (H >= Hrange(1) & H <= Hrange(2)) & (V >= Vrange(1) & V <= Vrange(2));

    I = rgb2gray(I);
    [h, w] = size(I);

     max_flower_radius = ceil(flowerRadius(2));
     middle_flower_radius = ceil((flowerRadius(2)+flowerRadius(1)) / 2);
     quarter_flower_radius = ceil((middle_flower_radius + flowerRadius(1)) / 2);
     min_flower_radius = ceil(flowerRadius(1));

    %% Use color channels to produce rough binary mask
    greenMinusBlue = G-B;
    greenMinusBlue = greenMinusBlue < 30;

    shadows = (G-B) > 10 & (G-B) < 30;
    shadows = imgaussfilt(double(shadows), 3) > 0.95;

    greenMinusRed = G-R;
    greenMinusRed = greenMinusRed > 0;

    disks = greenMinusBlue & ~greenMinusRed;

    %% Threshold gradient magnitude
    [Gmag, ~] = imgradient(G-B);
    edgeThreshold = Gmag > 20;

    % Thin thresholded edges
    se = strel('disk', 1);
    edgeThreshold = imerode(edgeThreshold, se);

    %% Remove "yellowish" pixels (i.e., branches and flower petals) from mask
    yellowish_pixels = imdilate(yellowish_pixels, se);
    disks_filtered = disks & ~yellowish_pixels;

    % Remove edges and 'shadows'
    disks_filtered = disks_filtered & ~edgeThreshold;
    disks_filtered(shadows) = 0;

    disks_filtered = imgaussfilt(double(disks_filtered), 3) > 0.3;

    %% Label binary regions and filter
    [L, numRegions] = bwlabel(disks_filtered);
    all_areas = accumarray(L(L > 0), 1, [numRegions 1]);

    stats = regionprops(L, 'Eccentricity', 'ConvexArea', 'Area');

    min_size = minROISize;
    ecc  = [stats.Eccentricity];
    ca   = [stats.ConvexArea];
    ar   = [stats.Area];
    orca = ca ./ ar;

    % Filter based on area, and other region statistics
    valid_indices = find(all_areas > min_size & ecc' <= maxROIEccentricity & orca' <= maxROIORCA)';
    filteredRegions = ismember(L, valid_indices);

    % For certain regions, replace with its convex hull
    [L, numRegions] = bwlabel(filteredRegions);
    convex_out = false(size(filteredRegions));
    for k = 1:numRegions
        regionMask = (L == k);
        if sum(regionMask(:)) <= sizeConvexHull(2) && sum(regionMask(:)) > sizeConvexHull(1)
            hull = bwconvhull(regionMask);
            convex_out = convex_out | hull;
        else
             convex_out = convex_out | regionMask;
        end
    end

    filledMask = imfill(convex_out, 'holes');

    greenish_pixels = 1-filledMask;

    %% Find circles in multiple passes with different parameters
    [centers_i1, radii_i1, centroids_i1] = dip_findfiltercircles(filledMask, [middle_flower_radius max_flower_radius], sensitivities(1), 0.4, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 1, 0, 1, 0, 0.6, 0, 1, -1, 0.1]);
    [centers_i1_merged, radii_i1_merged, centroids_i1_merged] = dip_mergecircles(centers_i1, radii_i1, centroids_i1, 0.6, 25);

    % Supress found circles in original image
    mask = dip_createcirclemask(centers_i1_merged, radii_i1_merged, 1.1, [h,w]);
    filledMask = filledMask & ~mask;

    [centers_i2, radii_i2, centroids_i2] = dip_findfiltercircles(filledMask, [min_flower_radius max_flower_radius-5], sensitivities(2), 1, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 0.5, 0, 1, 0, 1, 0, 1, 0.02, 1, -1, 1]);
    [centers_i2_merged, radii_i2_merged, centroids_i2_merged] = dip_mergecircles(centers_i2, radii_i2, centroids_i2, 0.6, 25);

    % Supress found circles in original image
    mask = dip_createcirclemask(centers_i2_merged, radii_i2_merged, 1.1, [h,w]);
    filledMask = filledMask & ~mask;

    [centers_i3, radii_i3, centroids_i3] = dip_findfiltercircles(filledMask, [min_flower_radius quarter_flower_radius], sensitivities(3), 1, V, filledMask, greenish_pixels, greenChannel-blueChannel, [20, 92, 92, 20], [0, 0.5, 0, 1, 0, 1, 0, 1, 0.02, 1, -1, 1]);

    centers_total = [centers_i1_merged; centers_i2_merged; centers_i3];
    radii_total   = [radii_i1_merged; radii_i2_merged; radii_i3];
    centroids_total = [centroids_i1_merged; centroids_i2_merged; centroids_i3];

    [centers, radii, ~] = dip_mergecircles(centers_total, radii_total, centroids_total, 0.6, 25);

end