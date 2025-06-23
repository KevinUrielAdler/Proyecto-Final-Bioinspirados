function [population, fitness, bestSolution, bestFitness] = Algorithm1(population, fitness, iter, params, evalFun, envParams, huboCambio)
    % Parámetros del PSO (según DefaultPSO)
    popSize = params.popSize;
    numVariables = params.numVariables;
    bounds = params.bounds;
    maximize = params.maximize;
    
    % Parámetros de PSO (valores fijos como en DefaultPSO)
    c1 = 0.05; % Aprendizaje cognitivo
    c2inicio = 1; % Aprendizaje social inicial
    c2final = 1.5; % Aprendizaje social final
    wm = randn(popSize, 1); % Inercia inicial
    n = zeros(popSize, 1); % Contador para inercia
    
    % Variables persistentes como en DefaultPSO
    persistent pbest pbestFitness velocities w_inercia mutationRates lastBestFitness
    
    % 1. Inicialización (primera iteración)
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
        mutationRates = 0.1 * ones(popSize, 1);
        lastBestFitness = min(fitness);
    end
    
    % Matriz topológica Double-Linked (como en DefaultPSO)
    Nbh = zeros(popSize, 3);
    Nbh(:, 1) = (1:popSize)';
    Nbh(:, 2) = (0:popSize-1)';
    Nbh(:, 3) = (2:popSize+1)';
    Nbh(1, 2) = popSize;
    Nbh(end, end) = 1;
    
    % 2b. Detección de cambios
    currentBest = min(fitness);
    if huboCambio
        fprintf('>>> Cambio detectado en iteración %d. Aumentando diversidad <<<\n', iter);
        
        % 2c. Aumentar diversidad
        mutationRates = min(mutationRates * 3, 0.5);
        for i = 1:popSize
            if rand() < 0.3 % Reubicar 30% de población
                population(i,:) = bounds(1,:) + rand(1,numVariables).*(bounds(2,:)-bounds(1,:));
                velocities(i,:) = zeros(1,numVariables);
            end
        end
    end
    
    % Determinar Gbest local (como en DefaultPSO)
    [gbest, gbestFitness] = localBestNBH(popSize, numVariables, Nbh, pbest, pbestFitness, maximize);
    
    % Actualizar posiciones (adaptado para incluir mutación)
    [population, velocities, w_inercia] = actualizarPosicionConMutacion(...
        population, velocities, wm, c1, c2inicio, c2final, popSize, ...
        numVariables, pbest, gbest, iter, params.numIteraciones, mutationRates, bounds);
    
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
    function [posiciones, velocidades, w] = actualizarPosicionConMutacion(...
            posicion, velocidad, wm, c1, c2inicio, c2final, Np, numVariables, ...
            pbest, gbest, it, maxIter, mutationRates, bounds)
        
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
                
                % Aplicar mutación
                if rand() < mutationRates(i)
                    mutation = 0.1*(2*rand-1)*(bounds(2,j)-bounds(1,j));
                    posiciones(i,j) = posiciones(i,j) + mutation;
                end
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