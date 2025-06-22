% Implementación del PSO para el framework modular
function [population, fitness, bestSolution, bestFitness] = DefaultPSO(population, fitness, iter, params, evalFun, envParams, huboCambio)

if huboCambio
    fprintf(">>> Cambio en el entorno en la iteración %d <<<\n", iter);
end

% Parámetros del PSO
popSize = params.popSize;
numVariables = params.numVariables;
bounds = params.bounds;
maximize = params.maximize;
c1 = 0.05; % Aprendizaje cognitivo
c2inicio = 1; % Aprendizaje social inicial
c2final = 1.5; % Aprendizaje social final
wm = randn(popSize, 1); % Inercia inicial
n = zeros(popSize, 1); % Contador para inercia

% Matriz topológica Double-Linked
Nbh = zeros(popSize, 3);
Nbh(:, 1) = (1:popSize)';
Nbh(:, 2) = (0:popSize-1)';
Nbh(:, 3) = (2:popSize+1)';
Nbh(1, 2) = popSize;
Nbh(end, end) = 1;

% Inicializar variables persistentes si es la primera iteración
persistent pbest pbestFitness velocities w_inercia;
if iter == 1

    % Asegurarnos que no se tomen en cuenta las areas donde no hay region
    if ~maximize
        for k = 1:popSize
            if isnan(fitness(k))
                fitness(k) = 1000000;
            end
        end
    else
        for k = 1:popSize
            if isnan(fitness(k))
                fitness(k) = -1000000;
            end
        end
    end

    pbest = population;
    pbestFitness = fitness;
    velocities = zeros(popSize, numVariables);
    for i = 1:popSize
        for j = 1:numVariables
            d = bounds(2, j) - bounds(1, j);
            velocities(i, j) = -d + 2 * rand * d;
        end
    end
    w_inercia = wm;
end

% Determinar Gbest local
[gbest, gbestFitness] = localBestNBH(popSize, numVariables, Nbh, pbest, pbestFitness, maximize);

% Actualizar posiciones
[population, velocities, w_inercia] = actualizarPosicion(population, velocities, wm, c1, c2inicio, c2final, popSize, numVariables, pbest, gbest, iter, params.numIteraciones);

% Rectificar restricciones
li = bounds(1, :);
ls = bounds(2, :);
for i = 1:popSize
    for j = 1:numVariables
        if population(i, j) < li(j) || population(i, j) > ls(j)
            population(i, j) = li(j) + rand * (ls(j) - li(j));
            d = ls(j) - li(j);
            velocities(i, j) = -d + 2 * rand * d;
        end
    end
end

% Evaluar nuevas posiciones
FOold = fitness;
for i = 1:popSize
    fitness(i) = evalFun(population(i, :), envParams);
    if maximize
        fitness(i) = -fitness(i); % Negativo para maximizar
    end
end

% Actualizar inercia
for k = 1:popSize
    if (maximize && fitness(k) > FOold(k)) || (~maximize && fitness(k) < FOold(k))
        wm(k) = (wm(k) * n(k) + w_inercia(k)) / (n(k) + 1);
        n(k) = n(k) + 1;
    end
end

% Actualizar mejores posiciones
for i = 1:popSize
    if (maximize && fitness(i) > pbestFitness(i)) || (~maximize && fitness(i) < pbestFitness(i))
        pbest(i, :) = population(i, :);
        pbestFitness(i) = fitness(i);
    end
end

% Determinar mejor solución global
bestFitness = min(pbestFitness);
bestSolution = pbest(find(pbestFitness == bestFitness, 1), :);

% Funciones auxiliares
    function [posiciones, velocidades, w] = actualizarPosicion(posicion, velocidad, wm, c1, c2inicio, c2final, Np, numVariables, pbest, gbest, it, maxIter)
        velocidades = zeros(Np, numVariables);
        posiciones = zeros(Np, numVariables);
        w = wm + 0.1 * randn(Np, 1);
        c2 = c2inicio + (c2final - c2inicio) * (it / maxIter);
        for i = 1:Np
            for j = 1:numVariables
                r1 = rand;
                r2 = rand;
                velocidades(i, j) = w(i) * velocidad(i, j) + c1 * r1 * (pbest(i, j) - posicion(i, j)) + ...
                    c2 * r2 * (gbest(i, j) - posicion(i, j));
                posiciones(i, j) = posicion(i, j) + velocidades(i, j);
            end
        end
    end

    function [gbest, gbestFitness] = localBestNBH(Np, numVariables, nbh, xpbest, FObest, maximize)
        gbest = zeros(Np, numVariables);
        gbestFitness = zeros(Np, 1);
        for i = 1:Np
            P1 = xpbest(nbh(i, 1), :);
            aptitudP1 = FObest(nbh(i, 1));
            P2 = xpbest(nbh(i, 2), :);
            aptitudP2 = FObest(nbh(i, 2));
            P3 = xpbest(nbh(i, 3), :);
            aptitudP3 = FObest(nbh(i, 3));
            cajaParticulas = [P1; P2; P3];
            cajaAptitudes = [aptitudP1; aptitudP2; aptitudP3];
            if maximize
                FOLbest = max(cajaAptitudes);
            else
                FOLbest = min(cajaAptitudes);
            end
            gbest(i, :) = cajaParticulas(find(cajaAptitudes == FOLbest, 1), :);
            gbestFitness(i) = FOLbest;
        end
    end
end
