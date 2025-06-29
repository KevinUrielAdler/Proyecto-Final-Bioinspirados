% Lista de algoritmos
algorithms = {'Algorithm1', 'Algorithm2', 'Algorithm3', 'Algorithm4', 'Algorithm5', 'Algorithm6'};

% Número de benchmarks y repeticiones
numBenchmarks = 5;
numRuns = 20;

% Estructura para guardar resultados
results = struct();

for bID = 1:numBenchmarks
    for a = 1:length(algorithms)
        algo = algorithms{a};

        allSolutions = [];
        allFitnesses = [];
        runTimes = [];

        for run = 1:numRuns
            tic;
            [solution, fitness] = runOptimizer(bID, algo);
            elapsedTime = toc;

            allSolutions = [allSolutions; solution];
            allFitnesses = [allFitnesses; fitness];
            runTimes = [runTimes; elapsedTime];
        end

        % Cálculos
        [bestFitness, bestIdx] = min(allFitnesses);
        bestSolution = allSolutions(bestIdx, :);

        [worstFitness, worstIdx] = max(allFitnesses);
        worstSolution = allSolutions(worstIdx, :);

        avgSolution = mean(allSolutions, 1);
        avgFitness = mean(allFitnesses);
        stdFitness = std(allFitnesses);
        avgTime = mean(runTimes);

        % Guardar en estructura
        results(bID).(algo).bestSolution = bestSolution;
        results(bID).(algo).bestFitness = bestFitness;

        results(bID).(algo).avgSolution = avgSolution;
        results(bID).(algo).avgFitness = avgFitness;

        results(bID).(algo).worstSolution = worstSolution;
        results(bID).(algo).worstFitness = worstFitness;

        results(bID).(algo).stdFitness = stdFitness;
        results(bID).(algo).avgTime = avgTime;
    end
end


% Mostrar resultados por algoritmo
for a = 1:length(algorithms)
    algo = algorithms{a};
    fprintf('\nResultados del %s\n', algo);
    fprintf('---------------------------------------------------------------------------------------------\n');
    fprintf('| ID | Best Fitness | Avg Fitness | Worst Fitness | Std Dev | Avg Time (s) |\n');
    fprintf('---------------------------------------------------------------------------------------------\n');

    for bID = 1:numBenchmarks
        data = results(bID).(algo);
        fprintf('| %2d | %12.6f | %11.6f | %13.6f | %7.4f | %13.4f |\n', ...
            bID, ...
            data.bestFitness, ...
            data.avgFitness, ...
            data.worstFitness, ...
            data.stdFitness, ...
            data.avgTime);
    end

    fprintf('---------------------------------------------------------------------------------------------\n');
end

