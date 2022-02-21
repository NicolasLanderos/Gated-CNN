# Deep-Learning-Project
Repository for the proyect of the Digital Design of Embedded Systems course

El proyecto esta desarrollado en base a tensorflow 2.3 y keras.

### La siguiente es una descripción de cada archivo:
   - Training.py: Contiene funciones para la carga de datos y generación de datasets
   - Nets.py: Contiene la implementación de cada una de las redes, capas de cuantización y capas de generación de error.
   - Quantization_And_Errors.py: Contiene funciones para el testeo de modelos en base a distintos errores y niveles de cuantización
   - Simulation.py: Contiene el modelado de los buffers y el flujo de datos para cada red.
   - Plots.py: Contiene funciones útiles para graficar resultados.
   - Jupyter Notebooks: Prueban una red específica para un set de datos especifico, siguiendo el siguiente pipeline: Entrenamiento->Cuantización->Simulación->Estudio de Errores en los buffer.
   - Errors: Carpeta con los errores (numéricos) probados para cada red.
   - Stats: Carpeta con el duty de cada buffer (numérico) para cada red.
   - Trained Weights: Carpeta con los pesos entrenados de cada red (solo se incluyen cuando su peso es tolerado por github).
   - Figures: Carpeta de resultados para cada red, dataset, peso entrenado, incluye a:
     - Quantization Experiments: Contiene una imagen con los resultados de accuracy y/o Loss de la red (probado en todo el conjunto de datos de prueba) en base al número de bits de las activaciones y pesos (independientemente).
     - Buffer Heatmaps: Contiene imágenes del duty del buffer considerando cada dirección de tamaño igual a las activaciones, además de un gráfico de duty promedio según posición del bit.
     - Errores: Contiene imágenes de la Accuracy y/o Loss para cada buffer, para cada sección del buffer, para distinto número de errores, además BoxPlots de estos.  Las secciones están determinadas de acuerdo al número de capas que las utilizan (por ejemplo la primera sección suele ser utilizada por todas las capas)
    
### Particularidades:
   - PilotNet:
     - Dataset Basado en [Udacity's Self-Driving Car Nanodegree](https://github.com/udacity/self-driving-car-sim)
     - Se testearon 3953 imágenes durante la simulación del duty
     - Cada error mostrado en la carpeta Errors es el promedio del error de 200 muestras para cada combinación de: (tipo de error, buffer, sección, numero de errors)
     - Se incluyo por separado una imagen de errores en la parte baja del buffer (primeras secciones), usando un número absoluto de errores en lugar a uno relativo (usado en secciones de mayor tamaño)
