function [population, fitness, bestSolution, bestFitness] = Algorithm2(population, fitness, iter, params, evalFun, envParams, huboCambio)
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
    persistent pbest pbestFitness velocities w_inercia diversityCounter replacementRate
    
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
        diversityCounter = 0;
        replacementRate = 0.2; % Tasa de reemplazo para diversidad
    end
    
    % Matriz topológica Double-Linked (como en DefaultPSO)
    Nbh = zeros(popSize, 3);
    Nbh(:, 1) = (1:popSize)';
    Nbh(:, 2) = (0:popSize-1)';
    Nbh(:, 3) = (2:popSize+1)';
    Nbh(1, 2) = popSize;
    Nbh(end, end) = 1;
    
    % 2b. Mantener diversidad (cada 5 iteraciones o tras cambio)
    if mod(iter,5) == 0 || huboCambio
        diversityCounter = diversityCounter + 1;
        fprintf('>>> Manteniendo diversidad en iteración %d <<<\n', iter);
        
        % Calcular distancias entre individuos
        distances = pdist(population);
        distanceMatrix = squareform(distances);
        avgDistances = mean(distanceMatrix, 2);
        
        % Reemplazar los menos diversos
        [~, idx] = sort(avgDistances);
        numToReplace = round(replacementRate * popSize);
        
        for i = 1:numToReplace
            % Crear nuevo individuo diverso
            newInd = bounds(1,:) + rand(1,numVariables).*(bounds(2,:)-bounds(1,:));
            
            % Reemplazar individuo menos diverso
            population(idx(i),:) = newInd;
            velocities(idx(i),:) = zeros(1,numVariables);
            pbest(idx(i),:) = newInd;
            pbestFitness(idx(i)) = evalFun(newInd, envParams);
        end
    end
    
    % Determinar Gbest local (como en DefaultPSO)
    [gbest, gbestFitness] = localBestNBH(popSize, numVariables, Nbh, pbest, pbestFitness, maximize);
    
    % Actualizar posiciones (mismo que DefaultPSO)
    [population, velocities, w_inercia] = actualizarPosicion(...
        population, velocities, wm, c1, c2inicio, c2final, popSize, ...
        numVariables, pbest, gbest, iter, params.numIteraciones);
    
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
    
    % Funciones auxiliares (idénticas a DefaultPSO)
    function [posiciones, velocidades, w] = actualizarPosicion(...
            posicion, velocidad, wm, c1, c2inicio, c2final, Np, numVariables, ...
            pbest, gbest, it, maxIter)
        
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