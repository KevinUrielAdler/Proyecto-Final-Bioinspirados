function f = disjoint_rastrigin(x, regions, regionCenters, penaltyValue)
    inRegion = false;
    minVal = Inf;
    for r = 1:size(regions, 1)
        if x(1) >= regions(r, 1) && x(1) <= regions(r, 2) && x(2) >= regions(r, 3) && x(2) <= regions(r, 4)
            inRegion = true;
            shifted = x - regionCenters(r, :);
            A = 10;
            val = A * numel(shifted) + sum(shifted.^2 - A * cos(2 * pi * shifted));
            if val < minVal
                minVal = val;
            end
        end
    end
    f = inRegion * minVal + (~inRegion) * penaltyValue;
end