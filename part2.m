function part2(inFilename, csvFilename, outLabelFilename)
% PART 2 – Detect and classify price & barcode tags using Hough + color
%
% Usage:
%   part2('ptag_b.png','g03_part2.csv','g03_part2_labels.png');
%
% CSV format:
%   <mask#, ul_row, ul_col, lr_row, lr_col, tag_type>
% where tag_type: 0 = barcode (white background), 1 = price (gray background)

    clc; close all;

    %% ---------------------------------------------------------------
    % 1) Load image
    % ---------------------------------------------------------------
    I = im2double(imread(inFilename));
    Igray = rgb2gray(I);
    [H, W] = size(Igray);

    % HSV for color-based cues (tags are bright + low saturation)
    Ihsv = rgb2hsv(I);
    S = Ihsv(:,:,2);
    V = Ihsv(:,:,3);

    figure; imshow(I);
    title('Original price / barcode tag image');

    %% ---------------------------------------------------------------
    % 2) Brightness-based candidate mask (loose)
    % ---------------------------------------------------------------
    Ieq = adapthisteq(Igray);          % local contrast enhancement

    seTop   = strel('disk', 15);
    Itophat = imtophat(Ieq, seTop);    % bright objects on darker background

    T = graythresh(Itophat);
    brightMask = Itophat > T;
    brightMask = brightMask | (Itophat > 0.6*T);  % loosen a bit

    brightMask = imclearborder(brightMask);

    %% ---------------------------------------------------------------
    % 2b) Edge-based candidate mask (for rectangular shapes)
    % ---------------------------------------------------------------
    E = edge(Ieq,'canny',0.12,1.5);            % slightly lower threshold

    E = imdilate(E, strel('rectangle',[3 3])); % thicken edges

    % Close gaps horizontally & vertically
    E = imclose(E, strel('rectangle',[5 15]));
    E = imclose(E, strel('rectangle',[15 5]));

    % Fill closed loops
    edgeFill = imfill(E,'holes');

    edgeFill = bwareaopen(edgeFill, 100);      % remove very small blobs
    edgeFill = imclearborder(edgeFill);

    figure; imshow(edgeFill);
    title('Edge-based candidate regions');

    %% ---------------------------------------------------------------
    % 2c) Color-based candidate mask (bright & low saturation)
    % ---------------------------------------------------------------
    tagColorMask = (S < 0.45) & (V > 0.55);    % white/gray-ish, quite bright
    tagColorMask = imopen(tagColorMask, strel('rectangle',[3 3]));
    tagColorMask = imclearborder(tagColorMask);

    figure; imshow(tagColorMask);
    title('Color-based candidate regions (bright, low-sat)');

    %% ---------------------------------------------------------------
    % 3) Combined candidate mask (VERY broad)
    % ---------------------------------------------------------------
    candMask = brightMask | edgeFill | tagColorMask;
    candMask = imopen(candMask, strel('rectangle',[3 3])); % light cleanup

    figure; imshow(candMask);
    title('Combined bright + edge + color candidate mask');

    %% ---------------------------------------------------------------
    % 4) Label raw components and measure basic properties
    % ---------------------------------------------------------------
    [Lraw, numRaw] = bwlabel(candMask);
    props = regionprops(Lraw, Igray, ...
        'Area','BoundingBox','Eccentricity','Extent');

    %% ---------------------------------------------------------------
    % 5) Hough-based filtering: keep regions that look roughly rectangular
    % ---------------------------------------------------------------
    L            = zeros(H, W, 'uint16');   % final label image
    labelCounter = 0;
    tagBB        = [];                      % Nx4 [x y w h]
    tagMean      = [];                      % Nx1 mean intensity

    for i = 1:numRaw
        A   = props(i).Area;
        ecc = props(i).Eccentricity;
        ext = props(i).Extent;
        bb  = props(i).BoundingBox;        % [x, y, width, height]
        w0  = bb(3);
        h0  = bb(4);
        ar  = w0 / max(h0, eps);           % aspect ratio (width/height)

        % --- VERY loose geometric pre-filter ------------------------ %%% TUNABLE
        if A < 1000 || A > 80000          % size: drop only really tiny/huge blobs
            continue;
        end

        if ext < 0.12                     % require just a little fill inside box
            continue;
        end

        if ecc > 0.999                    % reject almost-perfect lines only
            continue;
        end

        if ar < 0.5 || ar > 16            % very broad range of rectangles
            continue;
        end

        % --- Extract patch for Hough & photometric checks -----------
        x0 = round(bb(1));  y0 = round(bb(2));
        w1 = round(bb(3));  h1 = round(bb(4));

        x1 = max(1, x0);
        y1 = max(1, y0);
        x2 = min(W, x0 + w1 - 1);
        y2 = min(H, y0 + h1 - 1);

        patchGray  = Igray(y1:y2, x1:x2);
        patchMask  = (Lraw(y1:y2, x1:x2) == i);   % this component only
        patchEdges = edge(patchGray,'canny',0.10,1.5);

        patchEdges = patchEdges & patchMask;

        if nnz(patchEdges) < 20           % need some edges                    %%% TUNABLE
            continue;
        end

        %% ---------------- Hough transform on patch ------------------
        [Hh, theta, rho] = hough(patchEdges);
        if max(Hh(:)) == 0
            continue;
        end

        % detect up to 20 strong lines in this patch
        P = houghpeaks(Hh, 20, 'Threshold', 0.10*max(Hh(:)));          %%% TUNABLE

        lines = houghlines(patchEdges, theta, rho, P, ...
                           'FillGap', 15, ...                          %%% TUNABLE
                           'MinLength', max(4, round(min(size(patchEdges))/10))); %%%

        % Require at least one reasonably strong line structure
        if numel(lines) < 1                                            %%% TUNABLE
            continue;
        end

        %% ---------------- Photometric & color check -----------------
        mu       = mean(patchGray(:));            % brightness
        sigma    = std(patchGray(:));             % texture / variation
        patchSat = S(y1:y2, x1:x2);               % saturation
        meanSat  = mean(patchSat(:));

        % Tags are usually reasonably bright and fairly low-saturation.
        if mu < 0.10                               % allow slightly darker tags %%% TUNABLE
            continue;
        end

        % Allow plenty of texture; only reject extremely noisy patches.
        if sigma > 0.40                            % even more permissive       %%% TUNABLE
            continue;
        end

        % Very soft saturation check: reject only very vivid-colored blobs
        if meanSat > 0.50                          % allow more saturation      %%% TUNABLE
            continue;
        end

        %% ---------------- Accept as tag -----------------------------
        labelCounter = labelCounter + 1;

        % Fill bounding box with this label in final label image
        L(y1:y2, x1:x2) = labelCounter;

        tagBB(labelCounter,:)   = [x1, y1, x2-x1+1, y2-y1+1]; %#ok<AGROW>
        tagMean(labelCounter,1) = mu;                         %#ok<AGROW>
    end

    nTags = labelCounter;
    fprintf('Number of tags after broad Hough + color filtering: %d\n', nTags);

    %% ---------------------------------------------------------------
    % 6) Classify: barcode (0, white) vs price (1, gray)
    % ---------------------------------------------------------------
    tagType = zeros(nTags,1);    % default 0 = barcode

    if nTags > 0
        thr = 0.5*(min(tagMean) + max(tagMean));  % mid-point threshold
        % darker tags → gray price labels (1)
        tagType(tagMean < thr) = 1;
    end

    %% ---------------------------------------------------------------
    % 7) Save label image & overlay for sanity check
    % ---------------------------------------------------------------
    rgbLabel = label2rgb(L, 'jet', 'k', 'shuffle');
    imwrite(rgbLabel, outLabelFilename);

    figure; imshow(rgbLabel);
    title('Label image (one label per detected tag)');

    figure; imshow(I); hold on;
    for k = 1:nTags
        bb = tagBB(k,:);
        rectangle('Position', bb, 'EdgeColor','cyan','LineWidth',1);
    end
    title(sprintf('Detected potential tags (Hough + color): %d', nTags));
    hold off;

    %% ---------------------------------------------------------------
    % 8) Write CSV: mask#, ul_row, ul_col, lr_row, lr_col, tag_type
    % ---------------------------------------------------------------
    fid = fopen(csvFilename, 'w');
    if fid == -1
        error('Could not open CSV file: %s', csvFilename);
    end

    for k = 1:nTags
        bb = tagBB(k,:);             % [x1, y1, w, h]
        ul_col = bb(1);
        ul_row = bb(2);
        lr_col = bb(1) + bb(3) - 1;
        lr_row = bb(2) + bb(4) - 1;

        fprintf(fid, '%d,%d,%d,%d,%d,%d\n', ...
            k, ul_row, ul_col, lr_row, lr_col, tagType(k));
    end

    fclose(fid);
end
