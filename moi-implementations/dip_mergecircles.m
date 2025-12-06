function [centers_merged, radii_merged] = dip_mergecircles(centers, radii, centroids, reduction, distThreshold)


    N = length(radii);

    radii = radii .* reduction;
    
    % Build an adjacency matrix of overlaps
    overlapMatrix = false(N, N);
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
    G_matrix = graph(overlapMatrix);
    groups = conncomp(G_matrix);
    
    % Figure out which circles to delete
    toDelete = false(N,1);

    for g = unique(groups)
        idx = find(groups == g);
        % Skip groups without overlaps
        if numel(idx) < 2
            continue
        end

        diffs = centers(idx,:) - centroids(idx, :);
        dists = sqrt(sum(diffs.^2, 2));            

        [minDist, ix] = min(dists);

        if minDist < distThreshold
            winner = idx(ix);
        else
            winner = idx(1);
        end

        toDelete(setdiff(idx, winner)) = true;
    end

    radii = radii ./ reduction;

    % Remove them
    centers_merged = centers(~toDelete, :);
    radii_merged   = radii(~toDelete);
end
