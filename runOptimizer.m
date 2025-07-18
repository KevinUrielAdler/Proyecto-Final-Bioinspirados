% Codigo base para evaluar el algoritmo con los diferentes benchmarks
% dinamicos
function [bestSolution, bestFitness] =  runOptimizer(benchmarkID, algorithmName)
clc; close all; warning off all;

% Asegurar el acceso a las carpetas de los benchmarks y los algoritmos
addpath('benchmarks\', 'algorithms\');

% Parametros
numIteraciones = 20000; % Numero de iteraciones
pobSize = 100; % Tamaño de la poblacion
numVariables = 2; % Dimension de problema
umbralCambio = 0.1; % Umbral para considerar un cambio en la funcion
numSentinelas = 100; % Numero de puntos de control para detectar cambios

% Inicializar parametros del benchmark
switch benchmarkID
    case 1 % df1_sphere

        % Declaramos los pámetros óptimos para cada una de las funciones
        if strcmp(algorithmName,'Algorithm1')
            algoParams.c1 = 2.3;
            algoParams.c2 = 2.7;
            algoParams.w = 0.60;
        elseif strcmp(algorithmName,'Algorithm2')
            algoParams.c1 = 1.3;
            algoParams.c2 = 2.7;
            algoParams.w = 0.8;
        elseif strcmp(algorithmName,'Algorithm5')
            algoParams.c1 = 2;
            algoParams.c2 = 1;
            algoParams.w = 0.8;
        elseif strcmp(algorithmName,'Algorithm6')
            algoParams.c1 = 2.0;
            algoParams.c2inicio = 0.8;
            algoParams.c2final = 1.5;
            algoParams.wm_mean = 0.6;
        end


        dim = 2; % Numero de dimensiones
        bounds = [-5, 5]; % Limtes de la funcion
        numPeaks = 15; % Numero de picos
        k = 0.4; % Paso de cambio
        t0 = numIteraciones / 2; % Velocidad de cambio
        rng(1); % Velocidad de reproductibildiad
        % Los siguientes parametros son los de cambio establecidos
        envParams.h0 = 30 + 20 * rand(numPeaks, 1);
        envParams.h = envParams.h0;
        envParams.ch = 20 * (2 * rand(numPeaks, 1) - 1);
        envParams.w0 = 1 + 1.5 * rand(numPeaks, 1);
        envParams.w = envParams.w0;
        envParams.cw = 0.5 * (2 * rand(numPeaks, 1) - 1);
        envParams.c0 = bounds(1) + (bounds(2) - bounds(1)) * rand(numPeaks, dim);
        envParams.c = envParams.c0;
        envParams.cc = 2 * (rand(numPeaks, dim) - 0.5);
        li = [bounds(1), bounds(1)];
        ls = [bounds(2), bounds(2)];
        maximize = false;
        gridStep = 0.2;
        zMax = 70;
        evalFun = @(x, env) gaussian_mixture(x, env.h, env.w, env.c);
    case 2 % MPB_Griewank

        if strcmp(algorithmName,'Algorithm1')
            algoParams.c1 = 2;
            algoParams.c2 = 1.5;
            algoParams.w = 0.6;
        elseif strcmp(algorithmName,'Algorithm2')
            algoParams.c1 = 1.5;
            algoParams.c2 = 1.5;
            algoParams.w = 0.8;
        elseif strcmp(algorithmName,'Algorithm5')
            algoParams.c1 = 1.5;
            algoParams.c2 = 0.5;
            algoParams.w = 0.9;
        elseif strcmp(algorithmName,'Algorithm6')
            algoParams.c1 = 1.4;
            algoParams.c2inicio = 0.3;
            algoParams.c2final = 0.6;
            algoParams.wm_mean = 0.8;
        end

        dim = 2;
        bounds = [-50, 50];
        numPeaks = 20; % Numero de picos
        envParams.severity = 1.0; % Gravedad de cambio
        envParams.changeFrequency = 10; % Cada cuanto se realiza el cambio
        envParams.h = 30 + 40 * rand(numPeaks, 1);
        envParams.w = 1 + 4 * rand(numPeaks, 1);
        envParams.c = bounds(1) + (bounds(2) - bounds(1)) * rand(numPeaks, dim);
        li = [bounds(1), bounds(1)];
        ls = [bounds(2), bounds(2)];
        maximize = false;
        gridStep = 1;
        zMax = 70;
        evalFun = @(x, env) mpb_eval(x, env.h, env.w, env.c);
    case 3 % GDBG_sphere
        
        if strcmp(algorithmName,'Algorithm1')
            algoParams.c1 = 2.5;
            algoParams.c2 = 1.5;
            algoParams.w = 0.7;
        elseif strcmp(algorithmName,'Algorithm2')
            algoParams.c1 = 3;
            algoParams.c2 = 2;
            algoParams.w = 0.7;
        elseif strcmp(algorithmName,'Algorithm5')
            algoParams.c1 = 1.5;
            algoParams.c2 = 1.5;
            algoParams.w = 0.7;
        elseif strcmp(algorithmName,'Algorithm6')
            algoParams.c1 = 0.05;
            algoParams.c2inicio = 1;
            algoParams.c2final = 1.5;
            algoParams.wm_mean = 0.6;
        end

        dim = 2;
        bounds = [-5.12, 5.12];
        envParams.changeFrequency = 10; % Frecuencia de cambio
        envParams.rotationAngleStep = pi / 10; % Angulo de rotacion de la funcion
        envParams.translationStep = 0.5; % Paso de traslado de la funcion
        envParams.center = zeros(1, dim); % Centro de la funcion
        envParams.angle = 0;
        li = [bounds(1), bounds(1)];
        ls = [bounds(2), bounds(2)];
        maximize = false;
        gridStep = 0.2;
        zMax = 50;
        evalFun = @(x, env) rastrigin_gdbg(x, env.center, env.angle);
    case 4 % switching_rastrigin

        if strcmp(algorithmName,'Algorithm1')
            algoParams.c1 = 1.5;
            algoParams.c2 = 1.5;
            algoParams.w = 0.3;
        elseif strcmp(algorithmName,'Algorithm2')
            algoParams.c1 = 1.5;
            algoParams.c2 = 1.5;
            algoParams.w = 0.3;
        elseif strcmp(algorithmName,'Algorithm5')        
            algoParams.c1 = 1.2;
            algoParams.c2 = 1;
            algoParams.w = 0.85;
        elseif strcmp(algorithmName,'Algorithm6')
            algoParams.c1 = 0.05;
            algoParams.c2inicio = 1;
            algoParams.c2final = 1.5;
            algoParams.wm_mean = 0.6;
        end

        dim = 2; % Dimensiones de la funcion
        bounds = [-5.12, 5.12]; % Limites de la funcion
        envParams.changeInterval = 10; % Intervalo de cambio de la funcion
        % Establecer los cambios dentro de la funcion
        envParams.configs(1).A = 10; envParams.configs(1).center = [0,0];
        envParams.configs(2).A = 15; envParams.configs(2).center = [1.5, -1.5];
        envParams.configs(3).A = 5; envParams.configs(3).center = [-2, 2];
        envParams.A = envParams.configs.A;
        envParams.center = envParams.configs.center;
        li = [bounds(1), bounds(1)];
        ls = [bounds(2), bounds(2)];
        maximize = false;
        gridStep = 0.2;
        zMax = 100;
        evalFun = @(x, env) rastrigin_switching(x, env.A, env.center);

    case 5 % disjoint_rosenbrock

        if strcmp(algorithmName,'Algorithm1')
            algoParams.c1 = 1.8;
            algoParams.c2 = 2.2;
            algoParams.w = 0.4;
        elseif strcmp(algorithmName,'Algorithm2')
            algoParams.c1 = 1.3;
            algoParams.c2 = 1.7;
            algoParams.w = 0.8;
        elseif strcmp(algorithmName,'Algorithm5')
            algoParams.c1 = 2.2;
            algoParams.c2 = 0.3;
            algoParams.w = 0.95;
        elseif strcmp(algorithmName,'Algorithm6')
            algoParams.c1 = 0.05;
            algoParams.c2inicio = 1;
            algoParams.c2final = 1.5;
            algoParams.wm_mean = 0.6;
        end

        dim = 2;
        bounds = [-6, 6];
        envParams.changeInterval = 10; % Cada cuanto cambia
        envParams.penaltyValue = 100;
        % Regiones dentro del espacio de busqueda a explorar
        envParams.regions = [
            -6, -3, -6, -3;
            3, 6, -6, -3;
            -6, -3, 3, 6;
            3, 6, 3, 6
            ];
        numRegions = size(envParams.regions, 1);
        envParams.regionCenters = zeros(numRegions, dim);
        for i = 1:numRegions
            envParams.regionCenters(i,:) = [(envParams.regions(i,1) + ...
                envParams.regions(i, 2)) / 2, ...
                (envParams.regions(i,3) + envParams.regions(i,4)) / 2];
        end
        li = [bounds(1), bounds(1)];
        ls = [bounds(2), bounds(2)];
        maximize = false;
        gridStep = 0.2;
        zMax = envParams.penaltyValue;
        evalFun = @(x, env) disjoint_rastrigin(x, env.regions, env.regionCenters, env.penaltyValue);
    otherwise
        error('Benchmark invalido. Usa 1, 2, 3, 4 o 5.');
end

% Inicializar sentinelas
[xm, ym] = meshgrid(linspace(bounds(1), bounds(2), numSentinelas));
sentinelas = [xm(:), ym(:)];
fitnessPrevios = zeros(size(sentinelas,1), 1);

for k = 1:size(sentinelas,1)
    fitnessPrevios(k) = evalFun(sentinelas(k,:), envParams);
end

% Inicializar poblacion
population = zeros(pobSize, numVariables);
for i = 1:pobSize
    for j = 1:numVariables
        population(i, j) = li(j) + rand * (ls(j) - li(j));
    end
end

% Evaluar población inicial
fitness = zeros(pobSize, 1);
for i = 1:pobSize
    fitness(i) = evalFun(population(i,:), envParams);
    if maximize
        fitness(i) = -fitness(i); % Maximizar los fitness
    end
end

bestSolution = population(find(fitness == min(fitness), 1), :);
bestFitness = min(fitness);

% Párametros del algoritmo (se ajustan por algoritmo)
algoParams.popSize = pobSize;
algoParams.numIteraciones = numIteraciones;
algoParams.numVariables = numVariables;
algoParams.bounds = [li; ls];
algoParams.maximize = maximize;

% Cargar la funcion del algoritmo
try
    algoFunc = str2func(algorithmName);
catch
    error('Algoritmo %s no encontrado. Asegurate de que el archivo %s.m exista en la capeta de algorithms.', algorithmName, algorithmName);
end

% Visualizaicones (preparacion para ver la evolucion)
for it = 1:numIteraciones
    % Actualizar los parametros dinamicos de los entornos (benchmarks)
    switch benchmarkID
        case 1 % df1_sphere
            L = 1 ./ (1 + exp(-k * (it - t0)));
            envParams.h = envParams.h0 + envParams.ch * L;
            envParams.w = envParams.w0 + envParams.cw * L;
            envParams.c = envParams.c0 + envParams.cc * L;
        case 2 % MPB_Griewank
            if mod(it, envParams.changeFrequency) == 0
                [envParams.h, envParams.w, envParams.c] = update_peaks(envParams.h, envParams.w, envParams.c, bounds, envParams.severity);
                % fprintf(">>> Cambio de picos en iteracion %d <<<\n", it);
            end
        case 3 % GDBG_sphere
            if mod(it, envParams.changeFrequency) == 0
                envParams.angle = envParams.angle + envParams.rotationAngleStep;
                envParams.center = envParams.center + envParams.translationStep * (2 * rand(1, dim) - 1);
                % fprintf(">>> Cambio en el entorno en la iteracion %d <<<\n", it);
            end

        case 4 % switchin_rastrigin
            configIndex = ceil(it / envParams.changeInterval);
            if configIndex > length(envParams.configs)
                configIndex = length(envParams.configs);
            end
            envParams.A = envParams.configs(configIndex).A;
            envParams.center = envParams.configs(configIndex).center;

        case 5 % disjoint_rosenbrock
            if mod(it, envParams.changeInterval) == 0
                for i = 1:numRegions
                    x1 = rand * (envParams.regions(i,2) - envParams.regions(i,1)) + envParams.regions(i,1);
                    x2 = rand * (envParams.regions(i,4) - envParams.regions(i,3)) + envParams.regions(i,3);
                    envParams.regionCenters(i,:) = [x1, x2];
                end
                % fprintf(">>> Cambio de óptimos en iteración %d <<<\n", it);
            end
    end

    % Detectar cambios en el entorno
    fitnessActuales = zeros(size(sentinelas,1), 1);

    for k = 1:size(sentinelas,1)
        fitnessActuales(k) = evalFun(sentinelas(k,:), envParams);
    end

    delta = abs(fitnessActuales - fitnessPrevios) ./ max(abs(fitnessPrevios), 1e-8);
    delta = delta(~isnan(delta));
    cambioPromedio = mean(delta);
    huboCambio = cambioPromedio > umbralCambio;
    fitnessPrevios = fitnessActuales;

    % Ejecutar algoritmo bioinspirado
    [population, fitness, bestSolution, bestFitness] = algoFunc(population, fitness, it, algoParams, evalFun, envParams, huboCambio);

    % Visualizacion

    % Grid
    [X, Y] = meshgrid(bounds(1):gridStep:bounds(2));
    Z = zeros(size(X));
    surf(X, Y, Z, 'EdgeColor', 'none');
    colormap('jet'); % Cambiar colormap

    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            pt = [X(i,j), Y(i,j)];
            Z(i,j) = evalFun(pt, envParams);
        end
    end

    % Z = (Z - min(min(Z))) / (max(max(Z)) - min(min(Z))); % Normalizar Z

    clf;
    surf(X, Y, Z);
    hold on;
    scatter3(population(:, 1), population(:, 2), fitness, 50, 'r','filled');
    title(sprintf("benchmark %d con %s - iteracion %d", benchmarkID, algorithmName, it));
    xlabel('x_1'); ylabel('x_2'); zlabel('f(x)');
    axis([bounds(1) bounds(2) bounds(1) bounds(2) 0 zMax]);
    drawnow;

    disp(['Iteración ',num2str(it)]);
end

fprintf("Mejor solucion encontrada:");
disp(bestSolution);
fprintf("\nAptitud de la mejor solucion: %d", bestFitness);
