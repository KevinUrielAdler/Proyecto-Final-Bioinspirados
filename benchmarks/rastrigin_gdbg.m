function f = rastrigin_gdbg(x, center, angle)
    R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
    xTrans = R * (x(:) - center(:));
    A = 10;
    f = A * numel(xTrans) + sum(xTrans.^2 - A * cos(2 * pi * xTrans));
end