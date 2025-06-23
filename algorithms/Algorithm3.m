function [population, fitness, bestSolution, bestFitness] = GA_Multiploide(population, fitness, iter, params, evalFun, envParams, huboCambio)

if huboCambio
    fprintf(">>> Cambio en el entorno en la iteración %d <<<\n", iter);
end

persistent multiploides dominanceLevels;
popSize = params.popSize;
numVariables = params.numVariables;
bounds = params.bounds;
maximize = params.maximize;
mutationRate = 0.1;
crossoverRate = 0.8;
numAlleles = 3;

% Inicializar multiploides y niveles de dominancia
if iter == 1
    multiploides = zeros(popSize, numVariables, numAlleles);
    dominanceLevels = ones(popSize, numVariables, numAlleles) / numAlleles;

    for i = 1:popSize
        for j = 1:numVariables
            d = bounds(2, j) - bounds(1, j);

            for k = 1:numAlleles
                multiploides(i, j, k) = bounds(1, j) + rand * d;
            end
        end
    end
end

% Ajustar dominancia
if huboCambio
    dominanceLevels = dominanceLevels + 0.1 * randn(size(dominanceLevels));
    dominanceLevels = max(dominanceLevels, 0);
    s = sum(dominanceLevels, 3);

    for k = 1:numAlleles
        dominanceLevels(:, :, k) = dominanceLevels(:, :, k) ./ s;
    end
end

% Selección alelos dominantes
for i = 1:popSize
    for j = 1:numVariables
        probs = squeeze(dominanceLevels(i, j, :));
        idx = randsample(1:numAlleles, 1, true, probs);
        population(i, j) = multiploides(i, j, idx);
    end
end

% Cruza
numOffspring = round(popSize / 2);
offspring = zeros(numOffspring * 2, numVariables);

for i = 1:numOffspring
    p1 = randi(popSize);
    p2 = randi(popSize);

    for j = 1:numVariables
        if rand < crossoverRate
            alpha = rand;
            offspring(2*i-1, j) = alpha * population(p1, j) + (1 - alpha) * population(p2, j);
            offspring(2*i, j) = (1 - alpha) * population(p1, j) + alpha * population(p2, j);
        else
            offspring(2*i-1, j) = population(p1, j);
            offspring(2*i, j) = population(p2, j);
        end

        if rand < mutationRate
            d = bounds(2, j) - bounds(1, j);
            offspring(2*i-1, j) = offspring(2*i-1, j) + randn * 0.1 * d;
        end

        if rand < mutationRate
            d = bounds(2, j) - bounds(1, j);
            offspring(2*i, j) = offspring(2*i, j) + randn * 0.1 * d;
        end
    end
end

% Evaluación descendencia
offspringFitness = zeros(size(offspring, 1), 1);

for i = 1:size(offspring, 1)
    f = evalFun(offspring(i, :), envParams);
    offspringFitness(i) = maximize * -2 + 1 * f;
end

% Selección
combinedPop = [population; offspring];
combinedFit = [fitness; offspringFitness];
[~, idx] = sort(combinedFit);
population = combinedPop(idx(1:popSize), :);
fitness = combinedFit(idx(1:popSize));

% Actualizar multiploides
for i = 1:popSize
    for j = 1:numVariables
        worstIdx = randi(numAlleles);
        multiploides(i, j, worstIdx) = population(i, j);
    end
end

bestFitness = min(fitness);
bestSolution = population(find(fitness == bestFitness, 1), :);
end
