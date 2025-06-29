function [population, fitness, bestSolution, bestFitness] = Algorithm6(population, fitness, iter, algoParams, evalFun, envParams, huboCambio)

if huboCambio
    fprintf(">>> Cambio en el entorno en la iteración %d <<<\n", iter);
end

% Parámetros generales
popSize = algoParams.popSize;
numVariables = algoParams.numVariables;
bounds = algoParams.bounds;
maximize = algoParams.maximize;
numSubPop = 4;
fracPtrack = 0.5;
numTrack = round(numSubPop * fracPtrack);
numSearch = numSubPop - numTrack;

% Inicialización persistente
persistent Psearch Ptrack;
if iter == 1
    Psearch = initializeSubpops(algoParams, numSearch);
    Ptrack = initializeSubpops(algoParams, numTrack);
end

% 2a. Búsqueda de óptimos
for i = 1:length(Psearch)
    [Psearch{i}.population, Psearch{i}.fitness, ~, ~] = ...
        DefaultPSO(Psearch{i}.population, Psearch{i}.fitness, iter, algoParams, evalFun, envParams, huboCambio);
end

% 2b. Seguimiento de cambios
for i = 1:length(Ptrack)
    [Ptrack{i}.population, Ptrack{i}.fitness] = ...
        TrackSubpop(Ptrack{i}.population, Ptrack{i}.fitness, iter, algoParams, evalFun, envParams, huboCambio);
end

% 2c. Mantener diversidad (no implementado aún)

% 2d. Ajustar Psearch con experiencia de Ptrack
Psearch = adjustWithTrack(Psearch, Ptrack);

% 2f. Unir resultados
population = [];
fitness = [];
allBestFitness = [];
allBestSolutions = [];

for i = 1:length(Psearch)
    population = [population; Psearch{i}.population];
    fitness = [fitness; Psearch{i}.fitness];
    [minFit, idx] = min(Psearch{i}.fitness);
    allBestFitness(end+1) = minFit;
    allBestSolutions(end+1, :) = Psearch{i}.population(idx, :);
end

[bestFitness, idx] = min(allBestFitness);
bestSolution = allBestSolutions(idx, :);

if maximize
    bestFitness = -bestFitness;
    fitness = -fitness;
end

end


function subpops = initializeSubpops(params, num)
    popSize = params.popSize;
    numVariables = params.numVariables;
    bounds = params.bounds;

    subpops = cell(1, num);
    for i = 1:num
        population = rand(popSize, numVariables) .* ...
                     (bounds(2,:) - bounds(1,:)) + bounds(1,:);
        fitness = inf(popSize, 1);
        subpops{i} = struct('population', population, 'fitness', fitness);
    end
end

function subpops = adjustWithTrack(subpops, trackpops)
    for i = 1:length(trackpops)
        [~, idx] = min(trackpops{i}.fitness);
        elite = trackpops{i}.population(idx, :);
        j = mod(i-1, length(subpops)) + 1;
        subpops{j}.population(1, :) = elite;
    end
end

function [population, fitness] = TrackSubpop(population, fitness, iter, params, evalFun, envParams, huboCambio)
    [population, fitness] = DefaultPSO(population, fitness, iter, params, evalFun, envParams, huboCambio);
end