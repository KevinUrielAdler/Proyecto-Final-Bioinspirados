# Proyecto final

## Para ejecutar

1. Abrir la carpeta del proyecto en matlab.
2. Desde la ventana de comandos, ejecutar la función `runOptimizer(<benchmarkID>, '<algorithmName>')`.
    - `benchmarkID`: ID del benchmark a utilizar:
        - `1`: gaussian_mixture.
        - `2`: mpb_eval.
        - `3`: rastrigin_gdbg.
        - `4`: rastrigin_switching.
        - `5`: disjoint_rastrigin.
    - `algorithmName`: Nombre del algoritmo a utilizar (de la carpeta `./algorithms/`).

Ejemplo:

```bash
>> runOptimizer(4, 'DefaultPSO');
```

## Para Crear un nuevo algoritmo

### Estructura del archivo

Para crear un nuevo algoritmo, se debe crear un archivo dentro de `./algorithms/`, de preferencia con el formato `Algorithm<NúmeroAlgoritmo>.m`. Este archivo debe estar basado en el archivo `./algorithms/DefaultPSO.m`, cambiando únicamente lo necesario para adaptarlo al meta-algoritmo descrito en el paper.

### Detección de cambios de la función benchmark

En caso de que el meta-algoritmo requiera detectar cuándo sucede un cambio en el benchmark, se debe usar el parámetro booleano `huboCambio` recibido al inicio del algoritmo, por ejemplo en el algoritmo base se utiliza únicamente para imprimir un mensaje en la consola:

```matlab
1 % Implementación del meta-algoritmo 4 del paper sobre el algoritmo PSO
2 function [population, fitness, bestSolution, bestFitness] = DefaultPSO(population, fitness, iter, params, evalFun, envParams, huboCambio)
3
4 if huboCambio
5     fprintf(">>> Cambio en el entorno en la iteración %d <<<\n", iter);
6 end
...
```

<sub>Fragmento extraído de <code>./algorithms/DefaultPSO.m</code></sub>

### Comportamiento del código

Es importante notar que el codigo debe describir una sola iteración del algoritmo, pues la función `runOptimizer` se encargará de llamar a la función del algoritmo en cada iteración.

Aunque sí se puede utilizar el parámetro `iter` para saber en qué iteración se encuentra e implementar lógica diferente dependiendo de ésta. Por ejemplo, en el algoritmo base se inicializan las variables globales en la primera iteración:

```matlab
...
27 % Inicializar variables persistentes si es la primera iteración
28 persistent pbest pbestFitness velocities w_inercia;
29 if iter == 1
...
46    pbest = population;
47    pbestFitness = fitness;
48    velocities = zeros(popSize, numVariables);
49    for i = 1:popSize
50        for j = 1:numVariables
51            d = bounds(2, j) - bounds(1, j);
52            velocities(i, j) = -d + 2 * rand * d;
53        end
54    end
55    w_inercia = wm;
56 end
...
```

<sub>Fragmento extraído de <code>./algorithms/DefaultPSO.m</code></sub>

Sin embargo, hay que tener en cuenta que la inicialización y evaluación de la población se realiza en la función `runOptimizer`, por lo que no es necesario repetir ese proceso.
