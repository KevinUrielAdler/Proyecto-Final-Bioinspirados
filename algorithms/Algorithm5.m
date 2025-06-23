function [population, fitness, bestSolution, bestFitness] = Algorithm5(population, fitness, iter, params, evalFun, envParams, huboCambio)

% --------------------------- Parámetros ---------------------------
popSize = params.popSize;
numVariables = params.numVariables;
bounds = params.bounds;
maximize = params.maximize;
c1 = 1;
c2 = 2;
w = 0.8;

% --------------------------- Variables persistentes ---------------------------
persistent pbest pbestFitness velocities historialDesplazamientos modeloRegresion ultimaPosicionOptima historialIteraciones

if iter == 1
    pbest = population;
    pbestFitness = fitness;
    velocities = randn(popSize, numVariables);
    historialDesplazamientos = [];
    historialIteraciones = [];
    ultimaPosicionOptima = mean(pbest, 1);  % Primer óptimo estimado
    modeloRegresion = [];
end

% --------------------------- Predicción del nuevo óptimo ---------------------------
if huboCambio && ~isempty(historialDesplazamientos)
    fprintf(">>> Cambio detectado en iteración %d. Estimando nuevo óptimo...\n", iter);

    if ~isempty(modeloRegresion)
        entrada = [iter, historialDesplazamientos(end, :)];
        deltaOptimo = zeros(1, numVariables);
        for v = 1:numVariables
            deltaOptimo(v) = predict(modeloRegresion{v}, entrada);
        end
        nuevaEstimacion = ultimaPosicionOptima + deltaOptimo;

        for i = 1:round(popSize / 3)
            perturbacion = randn(1, numVariables) * 0.05;
            population(i, :) = nuevaEstimacion + perturbacion;
        end
    end
end


% --------------------------- Actualizar posiciones PSO ---------------------------
gbest = determinarGbest(pbest, pbestFitness, maximize);

for i = 1:popSize
    r1 = rand; r2 = rand;
    velocities(i, :) = w * velocities(i, :) + ...
                       c1 * r1 * (pbest(i, :) - population(i, :)) + ...
                       c2 * r2 * (gbest - population(i, :));
    population(i, :) = population(i, :) + velocities(i, :);
end

% Restringir dentro de los límites
li = bounds(1, :); ls = bounds(2, :);
for i = 1:popSize
    for j = 1:numVariables
        if population(i, j) < li(j) || population(i, j) > ls(j)
            population(i, j) = li(j) + rand * (ls(j) - li(j));
        end
    end
end

% --------------------------- Evaluación ---------------------------
for i = 1:popSize
    fitness(i) = evalFun(population(i, :), envParams);
end
if maximize 
    fitness = -fitness; 
end

% --------------------------- Actualizar pbest ---------------------------
for i = 1:popSize
    if (maximize && fitness(i) > pbestFitness(i)) || (~maximize && fitness(i) < pbestFitness(i))
        pbest(i, :) = population(i, :);
        pbestFitness(i) = fitness(i);
    end
end

% --------------------------- Actualizar modelo si hubo cambio ---------------------------
[bestFitness, idx] = min(pbestFitness);
bestSolution = pbest(idx, :);

if huboCambio
    desplazamiento = bestSolution - ultimaPosicionOptima;
    historialDesplazamientos = [historialDesplazamientos; desplazamiento];
    historialIteraciones = [historialIteraciones; iter];
    ultimaPosicionOptima = bestSolution;

    if size(historialDesplazamientos, 1) >= 3
        X = [historialIteraciones(1:end-1), historialDesplazamientos(1:end-1, :)];
        Y = historialDesplazamientos(2:end, :);

        modeloRegresion = cell(1, numVariables);  % modelo por variable
        for v = 1:numVariables
            modeloRegresion{v} = fitlm(X, Y(:, v));  % modelo para la dimensión v
        end
    end
end


% --------------------------- Función auxiliar ---------------------------
function gbest = determinarGbest(pbest, pbestFitness, maximize)
    if maximize
        [~, idx] = max(pbestFitness);
    else
        [~, idx] = min(pbestFitness);
    end
    gbest = pbest(idx, :);
end

end
