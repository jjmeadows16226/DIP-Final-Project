function part1(inFilename, csvFilename, outOverlayFilename)
% part1_darkCenters
% Detects flowers by finding their dark centers, draws circles for the full
% flowers on top of the original image, saves that overlay, builds a label
% image, and writes a CSV with center coordinates and radii.
%
% INPUTS:
%   inFilename       - string, path to input color image
%   csvFilename      - string, path to output CSV file
%   outOverlayFilename - string, path to save the overlay image (with circles)
%
% OUTPUT FILES:
%   1) outOverlayFilename: PNG/JPG of original image + detected flower circles
%   2) 'g03_part1_labels.png': label image, each flower region has a unique ID
%   3) csvFilename: CSV with rows: [flowerID, rowCenter, colCenter, radius]

    clc; close all;  % Clear command window and close figures (optional convenience)

    %% --- Load image and basic setup ---
    I = im2double(imread(inFilename));  % I: input RGB image in [0,1]
    Igray = rgb2gray(I);                % Igray: grayscale version used for detection
    [H, W] = size(Igray);               % H: image height (rows), W: image width (cols)

    figure;
    imshow(I);
    title('Original Image');            % Just for visualization, not used in processing


    %% --- Preprocessing for dark center detection ---
    % Apply a small Gaussian blur to reduce noise and texture.
    % 1.2 = standard deviation (sigma) of Gaussian, in pixels;
    % larger sigma -> more smoothing, might blur details.
    Iblur = imgaussfilt(Igray, 1.2);

    % Canny edge detection to emphasize boundaries of dark centers.
    % BWedge is a binary edge map (1 = edge, 0 = non-edge).
    BWedge = edge(Iblur, 'canny');


    %% --- Detect DARK CIRCLES ONLY with imfindcircles ---
    % rMin and rMax are the minimum and maximum circle radii (in pixels)
    % that we search for. They have a big impact on what gets detected.
    %
    % rMin = 15  -> ignore circles smaller than ~15 pixels radius (15 has
    % excess false positives, and minimal false negatives. Need to filter out false positive results)
    % rMax = 70  -> ignore circles larger than ~70 pixels radius (70 has
    % minimal false negatives due to flower size.)
    %
    % These values should roughly match the sizes of dark centers in the image.
    rMin = 15;
    rMax = 60;

    % imfindcircles looks for circular shapes in the edge image.
    % 'ObjectPolarity','dark'  -> we are looking for dark circles on a lighter background
    % 'Sensitivity',0.92       -> higher = more detections (including weaker ones)
    %                             lower = fewer, but stronger detections only.
    %                             0.92 seems to be the best value, along
    %                             with 0.12 for edge threshold based on
    %                             trial and error. May be wrong
    % 'EdgeThreshold',0.12     -> lower = more edge pixels considered;
    %                             helps find faint edges but can increase false positives.
    [centers, radii] = imfindcircles( ...
        BWedge, [rMin rMax], ...
        'ObjectPolarity', 'dark', ...
        'Sensitivity', 0.90, ...
        'EdgeThreshold', 0.12);

    fprintf('Raw dark-circle detections: %d\n', size(centers,1));


    %% --- Filter detections by center intensity (darkness check) ---
    % We keep only those circles whose interior is sufficiently dark in the
    % original grayscale image. This helps reject false positives.

    keep = true(size(centers,1), 1);    % keep(k) = true means detection k is kept

    for k = 1:size(centers,1)
        % center coordinates of circle k (x = column, y = row)
        cx = round(centers(k,1));
        cy = round(centers(k,2));
        r  = round(radii(k));           % radius of the detected dark center

        % If the center is somehow outside the image, discard it.
        if cx < 1 || cx > W || cy < 1 || cy > H
            keep(k) = false;
            continue;
        end

        % Extract a square patch around the center (roughly the center region
        % of the flower) to measure darkness.
        rr = max(1, cy - r) : min(H, cy + r);  % row range (clamped to image)
        cc = max(1, cx - r) : min(W, cx + r);  % column range (clamped)
        patch = Igray(rr, cc);

        % centerDarkness is the average intensity in that patch.
        % 0   = completely black
        % 1   = completely white
        centerDarkness = mean(patch(:));

        % Threshold for how dark the center must be.
        % 0.35 chosen empirically:
        %   - lower value = require darker centers (may reject some true centers)
        %   - higher value = allow lighter centers (may introduce more false positives)
        if centerDarkness > 0.25
            keep(k) = false;            % not dark enough -> discard this detection
        end
    end

    % Keep only the detections that passed the darkness test.
    centers = centers(keep, :);
    radii   = radii(keep);

    fprintf('Filtered dark-center detections: %d\n', size(centers,1));

        %% --- Additional filter: require yellow petals around the dark center ---

    % Define YELLOW detection thresholds (tweakable)
    yellowMinR = 0.30;   % minimum red channel intensity
    yellowMinG = 0.30;   % minimum green channel intensity
    yellowMaxB = 0.70;   % maximum blue channel intensity (yellow has low blue)

    % If you want to use HSV instead:
    % yellowHueMin = 0.10;  
    % yellowHueMax = 0.20;

    keep2 = true(size(centers,1),1);

    for k = 1:size(centers,1)

        cx = round(centers(k,1));
        cy = round(centers(k,2));
        r  = round(radii(k));

        % --- Sample a ring around the dark center ---
        ringInner = round(r * 1.2);     % just outside the black center
        ringOuter = round(r * 2.2);     % inside the yellow petal region

        % Build a ring mask
        rr = max(1, cy - ringOuter) : min(H, cy + ringOuter);
        cc = max(1, cx - ringOuter) : min(W, cx + ringOuter);

        [XX,YY] = meshgrid(cc, rr);
        d2 = (XX - cx).^2 + (YY - cy).^2;

        ringMask = (d2 >= ringInner^2) & (d2 <= ringOuter^2);

        R = I(rr,cc,1);
        G = I(rr,cc,2);
        B = I(rr,cc,3);

        % Identify yellow pixels via RGB criteria
        yellowMask = (R >= yellowMinR) & (G >= yellowMinG) & (B <= yellowMaxB);

        percentYellow = 100 * sum(yellowMask(:) & ringMask(:)) / sum(ringMask(:));

        % Require at least 8–15% yellow in the ring (tweakable)
        if percentYellow < 10
            keep2(k) = false;
        end
    end

    centers = centers(keep2,:);
    radii   = radii(keep2);

    fprintf('After yellow-petal check: %d\n', size(centers,1));

    %% --- Estimate full flower radius from center radius ---
    % fullRadii is used for drawing big circles that represent the whole flower
    % around the dark center.
    %
    % 3.0 is a scale factor:
    %   full flower radius ≈ 3 × dark center radius.
    % You can tweak this factor to grow/shrink the drawn flower circles.
    fullRadii = radii * 3.0;


    %% --- Display final detections on top of the original image ---
    figure;
    imshow(I);
    hold on;

    % Draw the estimated flower circles on the image:
    %   centers   -> circle centers (x,y)
    %   fullRadii -> circle radii (in pixels)
    % 'Color','cyan'      -> displayed color of the circles in the overlay
    % 'LineWidth',2       -> thickness of the circle outline in pixels
    viscircles(centers, fullRadii, 'Color', 'cyan', 'LineWidth', 2);

    title(sprintf('Detected Flowers (Dark-Center Method): %d', size(centers,1)));
    hold off;

    % Save the overlay figure (this is your "final image" with detections drawn)
    saveas(gcf, outOverlayFilename);


    %% --- Build label image (per-pixel flower ID) ---
    % L is an image where each pixel value tells you which flower it belongs to.
    % 0 = background (no flower), 1 = first flower, 2 = second flower, etc.
    L = zeros(H, W, 'uint16');

    % X,Y give the column (x) and row (y) indices for every pixel.
    % X(row,col) = column index
    % Y(row,col) = row index
    [X, Y] = meshgrid(1:W, 1:H);

    for k = 1:size(centers,1)
        cx = centers(k,1);   % center x (column)
        cy = centers(k,2);   % center y (row)
        r  = fullRadii(k);   % full flower radius for labeling

        % Create a circular mask for flower k:
        % (X - cx)^2 + (Y - cy)^2 <= r^2 describes a filled circle.
        mask = (X - cx).^2 + (Y - cy).^2 <= r.^2;

        % Assign label k to all pixels inside the circle.
        L(mask) = k;
    end

    % Convert label image to an RGB visualization:
    % 'jet'    -> colormap for different labels
    % 'k'      -> background color for label 0 (black)
    % 'shuffle'-> randomize color assignment for each label
    imwrite(label2rgb(L, 'jet', 'k', 'shuffle'), 'g03_part1_labels.png');


    %% --- Write CSV with center coordinates and radii ---
    % CSV format: flowerIndex, rowCenter(y), colCenter(x), radius
    % rowCenter and colCenter are rounded to integer pixel locations.
    
    fid = fopen(csvFilename, 'w');
    for k = 1:size(centers,1)
        cy = round(centers(k,2));        % row index of center
        cx = round(centers(k,1));        % column index of center
        r  = round(fullRadii(k));        % full flower radius
        fprintf(fid, '%d,%d,%d,%d\n', k, cy, cx, r);
    end
    fclose(fid);
    

end
