function [population, fitness, bestSolution, bestFitness] = Algorithm2(population, fitness, iter, params, evalFun, envParams, huboCambio)
    % Parámetros del PSO
    popSize = params.popSize;
    numVariables = params.numVariables;
    bounds = params.bounds;
    maximize = params.maximize;
    
    % Coeficientes de aprendizaje
    c1 = params.c1; % Cognitivo
    c2 = params.c2; % Social
    w = params.w;  % Inercia
    
    % Parámetros de diversidad
    diversityThreshold = 0.1 * (bounds(2,1) - bounds(1,1)); % 10% del rango
    replacementRate = 0.1; % 10% de la población se reemplaza en cada iteración
    
    % Inicializar variables persistentes
    persistent pbest pbestFitness velocities;
    if iter == 1 || isempty(pbest)
        % Manejar valores NaN en el fitness
        if ~maximize
            fitness(isnan(fitness)) = 1000000;
        else
            fitness(isnan(fitness)) = -1000000;
        end
        
        pbest = population;
        pbestFitness = fitness;
        velocities = zeros(popSize, numVariables);
        
        % Inicializar velocidades aleatorias dentro de los límites
        for i = 1:popSize
            for j = 1:numVariables
                d = bounds(2,j) - bounds(1,j);
                velocities(i,j) = -d + 2*rand*d;
            end
        end
    end
    
    % Mantener diversidad: Reemplazar partículas muy cercanas
    numReplace = round(replacementRate * popSize);
    distances = pdist2(population, population);
    distances(logical(eye(popSize))) = Inf; % Ignorar diagonal
    
    for k = 1:numReplace
        % Encontrar el par más cercano
        [minDist, idx] = min(distances(:));
        [i, j] = ind2sub(size(distances), idx);
        
        if minDist < diversityThreshold
            % Reemplazar una de las partículas con una nueva aleatoria
            for d = 1:numVariables
                population(j,d) = bounds(1,d) + rand*(bounds(2,d) - bounds(1,d));
                velocities(j,d) = -1 + 2*rand;
            end
            fitness(j) = evalFun(population(j,:), envParams);
            if maximize
                fitness(j) = -fitness(j);
            end
            pbest(j,:) = population(j,:);
            pbestFitness(j) = fitness(j);
            
            % Actualizar matriz de distancias
            distances(j,:) = pdist2(population(j,:), population);
            distances(:,j) = distances(j,:)';
            distances(j,j) = Inf;
        end
    end
    
    % Encontrar el mejor global (gbest)
    if maximize
        [gbestFitness, gbestIdx] = max(pbestFitness);
    else
        [gbestFitness, gbestIdx] = min(pbestFitness);
    end
    gbest = pbest(gbestIdx,:);
    
    % Actualizar velocidades y posiciones
    for i = 1:popSize
        for j = 1:numVariables
            r1 = rand;
            r2 = rand;
            
            % Ecuación de velocidad del PSO
            velocities(i,j) = w*velocities(i,j) + ...
                c1*r1*(pbest(i,j) - population(i,j)) + ...
                c2*r2*(gbest(j) - population(i,j));
            
            % Actualizar posición
            population(i,j) = population(i,j) + velocities(i,j);
            
            % Aplicar límites
            if population(i,j) < bounds(1,j)
                population(i,j) = bounds(1,j);
                velocities(i,j) = -0.5*velocities(i,j); % Rebote
            elseif population(i,j) > bounds(2,j)
                population(i,j) = bounds(2,j);
                velocities(i,j) = -0.5*velocities(i,j); % Rebote
            end
        end
        
        % Evaluar nueva posición
        newFitness = evalFun(population(i,:), envParams);
        if maximize
            newFitness = -newFitness;
        end
        
        % Actualizar mejor personal (pbest)
        if (maximize && newFitness > pbestFitness(i)) || (~maximize && newFitness < pbestFitness(i))
            pbest(i,:) = population(i,:);
            pbestFitness(i) = newFitness;
        end
        
        fitness(i) = newFitness;
    end
    
    % Determinar mejor solución global
    if maximize
        [bestFitness, bestIdx] = max(pbestFitness);
    else
        [bestFitness, bestIdx] = min(pbestFitness);
    end
    bestSolution = pbest(bestIdx,:);
    
    % Para maximización, convertir el fitness de vuelta a positivo
    if maximize
        bestFitness = -bestFitness;
    end
end
