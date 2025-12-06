function [centers, radii, cOM] = dip_findfiltercircles(I, radiusRange, sensitivity, scale, patchImage, BW, backgroundBW, grayscale, THRESH, COND)
    FULLN = 8;                         % total number of conditions expected
    DEFAULTS = [0 1 0 1 0 1 0 1];      % values that make every condition TRUE
                                       % (min=0, max=1 for ratios)
    
    if nargin < 10 || isempty(COND)
        COND = DEFAULTS;
    elseif numel(COND) < FULLN
        COND = [COND(:).'  DEFAULTS(numel(COND)+1 : FULLN)];
    end    

    centers = [];
    radii = [];
    cOM = [];
    
    [h, w] = size(I);
    [cen, rad] = imfindcircles(I, radiusRange, "Sensitivity", sensitivity);

    for k = 1:numel(rad)
        cx = round(cen(k,1));
        cy = round(cen(k,2));
        r  = round(rad(k) .* scale);
    
        rowRange = max(1, cy - (r)) : min(h, cy + (r));
        colRange = max(1, cx - (r)) : min(w, cx + (r));

        patch = patchImage(rowRange, colRange);
        
        patch_region = BW(rowRange, colRange);
        mask_ratio = nnz(patch_region) / numel(patch);
    
        patch_region_green = backgroundBW(rowRange, colRange);
        greenish_ratio = nnz(patch_region_green) / numel(patch);
    
        petal_mask = grayscale(rowRange, colRange) < THRESH(1) | grayscale(rowRange, colRange) > THRESH(2);
    
        just_petals_mask = grayscale(rowRange, colRange) > THRESH(3) & (1-patch_region);
        just_petals = nnz(just_petals_mask);
    
        just_disk_mask = grayscale(rowRange, colRange) < THRESH(4);
        just_disks = nnz(just_disk_mask);

        if nnz(patch_region) == 0
            continue
        end
    
        stats = regionprops(patch_region, 'Centroid');
        regionCentroid = stats.Centroid;
        regionCentroid = regionCentroid + [colRange(1)-1, rowRange(1)-1];
    
        if (just_petals / numel(patch)) >= COND(1) && (just_petals / numel(patch)) <= COND(2) && ...
           (just_disks / numel(patch)) >= COND(3) && (just_disks / numel(patch)) <= COND(4) && ...
           greenish_ratio >= COND(5) && greenish_ratio <= COND(6) && ...
           mask_ratio >= COND(7) && mask_ratio <= COND(8)
            centers = [centers; cen(k,:)];
            radii = [radii; rad(k)];
            cOM = [cOM; regionCentroid];
        end
    end
end