function part2(inFilename, csvFilename, outLabelFilename)
% PART 2 – Detect and classify price & barcode tags
%
% Usage:
%   part2('ptag_b.png','g03_part2.csv','g03_part2_labels.png');
%
% CSV format (one line per detected tag):
%   <mask#, ul_row, ul_col, lr_row, lr_col, tag_type>
% where tag_type: 0 = barcode (white), 1 = price (gray)

    clc; close all;

    %==================================================================
    % 1) Load image
    %==================================================================
    I      = im2double(imread(inFilename));   % RGB image in [0,1]
    Igray  = rgb2gray(I);
    [H, W] = size(Igray);

    figure; imshow(I);
    title('Original price / barcode tag image');

    %==================================================================
    % 2) Color-based mask for white / gray tags (HSV + RGB)
    %    We want nearly neutral, bright rectangles.
    %==================================================================
    hsvI = rgb2hsv(I);
    S    = hsvI(:,:,2);
    V    = hsvI(:,:,3);

    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);

    % --- Tunable color thresholds ---
    satMax   = 0.40;  % low saturation (near gray)
    valMin   = 0.55;  % fairly bright
    rgbDelta = 0.10;  % channels within ±0.10 of each other

    neutralHSV  = (S < satMax) & (V > valMin);
    neutralRGB  = (abs(R-G) < rgbDelta) & ...
                  (abs(R-B) < rgbDelta) & ...
                  (abs(G-B) < rgbDelta);

    tagColorMask = neutralHSV & neutralRGB;

    % light morphology cleanup
    tagColorMask = imopen( tagColorMask,  strel('disk',2));
    tagColorMask = imclose(tagColorMask,  strel('disk',3));
    tagColorMask = bwareaopen(tagColorMask, 50);

    figure; imshow(tagColorMask);
    title('HSV+RGB tag color mask');

    %==================================================================
    % 3) Local rectangle support from edges
    %==================================================================
    edgeMask = edge(Igray, 'canny', [], 1.0);

    % Remove border-connected edges to avoid a giant central region
    edgeMask = imclearborder(edgeMask);

    % Slight dilation so edges join up (horizontal + vertical)
    edgeMask = imdilate(edgeMask, strel('line',3,0)) | ...
               imdilate(edgeMask, strel('line',3,90));

    % Fill closed edge loops → local "solid" shapes
    filledEdges = imfill(edgeMask, 'holes');

    figure; imshow(filledEdges);
    title('Filled edge regions (local rectangles)');

    %==================================================================
    % 4) Combine color + local rectangles → candidate mask
    %==================================================================
    candidateMask = filledEdges & tagColorMask;

    % remove tiny blobs and do light opening
    candidateMask = bwareaopen(candidateMask, 200);
    candidateMask = imopen(candidateMask, strel('rectangle',[3 5]));

    figure; imshow(candidateMask);
    title('Candidate tag regions after rectangle + color');

    %==================================================================
    % 5) Label and measure regions
    %==================================================================
    [Lraw, numRaw] = bwlabel(candidateMask);

    props = regionprops(Lraw, Igray, ...
        'Area',        ...
        'BoundingBox', ...
        'Extent',      ...
        'Solidity');

    % --- Geometric constraints (tags are mid-size, axis-aligned rectangles) ---
    minTagArea   = 600;   % reject very small blobs
    maxTagArea   = 60000;  % ignore huge product fronts

    minTagWidth  = 40;
    maxTagWidth  = 1000;
    minTagHeight = 60;
    maxTagHeight = 1000;

    minAspect    = 1.0;    % width / height; tags wider than tall
    maxAspect    = 8.0;

    minExtent    = 0.50;   % how full the bounding box is
    minSolidity  = 0.70;   % how convex / solid

    % --- Photometric constraints inside each bbox ---
    muGrayMin        = 0.55;  % tags are fairly bright
    sdGrayMax        = 0.18;  % roughly uniform
    fracColorMaskMin = 0.50;  % at least half the bbox is tag-colored

    %==================================================================
    % 6) Filter to get likely tags
    %==================================================================
    L            = zeros(H, W, 'uint16'); % final label image
    labelCounter = 0;
    tagBB        = [];                    % [x,y,w,h] per tag
    tagMeanV     = [];                    % mean V (brightness) per tag

    for i = 1:numRaw

        % ---------- Basic geometry ----------
        A   = props(i).Area;
        bb  = props(i).BoundingBox;  % [x y w h]
        ext = props(i).Extent;
        sol = props(i).Solidity;

        w0 = bb(3);
        h0 = bb(4);
        ar = w0 / max(h0, eps);      % width / height

        if A  < minTagArea   || A  > maxTagArea,    continue; end
        if w0 < minTagWidth  || w0 > maxTagWidth,   continue; end
        if h0 < minTagHeight || h0 > maxTagHeight,  continue; end
        if ar < minAspect    || ar > maxAspect,     continue; end
        if ext < minExtent,                         continue; end
        if sol < minSolidity,                       continue; end

        % ---------- Extract bbox safely ----------
        x0 = max(1, round(bb(1)));
        y0 = max(1, round(bb(2)));
        x1 = min(W, x0 + round(w0) - 1);
        y1 = min(H, y0 + round(h0) - 1);

        patchGray  = Igray(y0:y1, x0:x1);
        patchV     = V    (y0:y1, x0:x1);
        patchMask  = tagColorMask(y0:y1, x0:x1);

        muGray        = mean(patchGray(:));
        sdGray        = std (patchGray(:));
        fracColorMask = mean(patchMask(:));

        % ---------- Photometric checks ----------
        if muGray        < muGrayMin,        continue; end
        if sdGray        > sdGrayMax,        continue; end
        if fracColorMask < fracColorMaskMin, continue; end

        % ---------- Accept this region as a tag ----------
        labelCounter = labelCounter + 1;

        L(y0:y1, x0:x1) = labelCounter;

        tagBB(labelCounter,:)  = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]; %#ok<AGROW>
        tagMeanV(labelCounter) = mean(patchV(:));                    %#ok<AGROW>
    end

    nTags = labelCounter;
    fprintf('Number of potential tags after rectangle filter: %d\n', nTags);

    %==================================================================
    % 7) Classify: barcode (0, white) vs price (1, gray)
    %==================================================================
    tagType = zeros(nTags, 1);  % default 0 = barcode

    if nTags > 0
        thr = 0.5 * (min(tagMeanV) + max(tagMeanV));   % simple mid-threshold
        % darker tags → gray price labels (1)
        tagType(tagMeanV < thr) = 1;
    end

    %==================================================================
    % 8) Save label image & overlay
    %==================================================================
    rgbLabel = label2rgb(L, 'jet', 'k', 'shuffle');
    imwrite(rgbLabel, outLabelFilename);

    figure; imshow(rgbLabel);
    title('Label image (one label per detected tag)');

    figure; imshow(I); hold on;
    for k = 1:nTags
        rectangle('Position', tagBB(k,:), ...
                  'EdgeColor', 'cyan', ...
                  'LineWidth', 1);
    end
    title(sprintf('Detected potential tags (rectangle search): %d', nTags));
    hold off;

    %==================================================================
    % 9) Write CSV: mask#, ul_row, ul_col, lr_row, lr_col, tag_type
    %==================================================================
    fid = fopen(csvFilename, 'w');
    if fid == -1
        error('Could not open CSV file: %s', csvFilename);
    end

    for k = 1:nTags
        bb = tagBB(k,:);        % [x1,y1,w,h]
        ul_col = bb(1);
        ul_row = bb(2);
        lr_col = bb(1) + bb(3) - 1;
        lr_row = bb(2) + bb(4) - 1;

        fprintf(fid, '%d,%d,%d,%d,%d,%d\n', ...
                k, ul_row, ul_col, lr_row, lr_col, tagType(k));
    end

    fclose(fid);
end
