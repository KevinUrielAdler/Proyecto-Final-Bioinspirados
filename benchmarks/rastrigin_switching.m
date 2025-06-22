function f = rastrigin_switching(x, A, center)
    x = x - center;
    f = A * numel(x) + sum(x.^2 - A * cos(2 * pi * x));
end