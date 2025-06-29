function [population, fitness, bestSolution, bestFitness] = Algorithm1(population, fitness, iter, params, evalFun, envParams, huboCambio)
    % Parámetros del PSO
    popSize = params.popSize;
    numVariables = params.numVariables;
    bounds = params.bounds;
    maximize = params.maximize;
    
    % Coeficientes de aprendizaje
    c1 = params.c1; % Cognitivo
    c2 = params.c2; % Social
    w = params.w;  % Inercia
    
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
    
    % Si hubo cambio, aumentar diversidad
    if huboCambio
        fprintf(">>> Cambio detectado - Reinicializando diversidad en iteración %d <<<\n", iter);
        
        % Reinicializar el 30% de la población aleatoriamente
        numReinit = round(0.3*popSize);
        reinitIdx = randperm(popSize, numReinit);
        
        for i = reinitIdx
            for j = 1:numVariables
                population(i,j) = bounds(1,j) + rand*(bounds(2,j) - bounds(1,j));
                velocities(i,j) = -1 + 2*rand; % Nueva velocidad aleatoria
            end
            fitness(i) = evalFun(population(i,:), envParams);
            if maximize
                fitness(i) = -fitness(i);
            end
        end
        
        % Actualizar pbest para las partículas reinicializadas
        for i = reinitIdx
            pbest(i,:) = population(i,:);
            pbestFitness(i) = fitness(i);
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
