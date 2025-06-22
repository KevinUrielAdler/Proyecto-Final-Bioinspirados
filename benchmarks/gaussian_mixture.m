function val = gaussian_mixture(x, h, w, c)
    m = size(c, 1);
    vals = zeros(m, 1);
    for i = 1:m
        dist = norm(x - c(i, :));
        vals(i) = h(i) * exp(- (dist^2) / (2 * w(i)^2));
    end
    val = max(vals);
end