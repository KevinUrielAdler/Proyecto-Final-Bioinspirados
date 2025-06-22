function [h, w, c] = update_peaks(h, w, c, bounds, severity)
    delta = severity * (2 * rand(size(c)) - 1);
    c = c + delta;
    c = max(min(c, bounds(2)), bounds(1));
    h = h + 2 * (rand(size(h)) - 0.5);
    w = w + 0.1 * (rand(size(w)) - 0.5);
    w = max(w, 0.5);
end