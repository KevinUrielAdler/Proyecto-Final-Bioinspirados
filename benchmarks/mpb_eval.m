function val = mpb_eval(x, h, w, c)
    numPeaks = size(h, 1);
    values = zeros(numPeaks, 1);
    for i = 1:numPeaks
        dist = norm(x - c(i, :));
        values(i) = h(i) * exp(- (dist^2) / (2 * w(i)^2));
    end
    val = max(values);
end