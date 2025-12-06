function mask = dip_createcirclemask(centers, radii, scale, sz)
    mask = false(sz(1), sz(2));
    
    for k = 1:size(centers,1)
        cx = round(centers(k,1));
        cy = round(centers(k,2));
        r  = round(radii(k) .* scale);
    
        rowRange = max(1, cy - r) : min(sz(1), cy + r);
        colRange = max(1, cx - r) : min(sz(2), cx + r);
        % mask(rowRange, colRange) = true;
 
        dot = false(length(rowRange), length(colRange));
        dot(ceil(end/2), ceil(end/2)) = true;
        mask(rowRange, colRange) = mask(rowRange, colRange) | bwdist(dot) <= radii(k) .* scale;
    end
end