# Deep-Learning-Proyect
Repository for the proyect of the Digital Design of Embedded Systems course

El proyecto esta desarrollado en base a tensorflow 2.3 y keras.

### La siguiente es una descripcion de cada archivo:
   - Training.py: Contiene funciones para la carga de datos y generacion de datasets
   - Nets.py: Contiene la implementacion de cada una de las redes, capas de cuantizacion y capas de generacion de error.
   - Quantization_And_Errors.py: Contiene funciones para el testeo de modelos en base a distintos errores y niveles de cuantizacion
   - Simulation.py: Contiene el modelado de los buffers y el flujo de datos para cada red.
   - Plots.py: Contiene funciones utiles para graficar resultados.
   - Jupyter Notebooks: Prueban una red especifica para un set de datos especifico, siguiendo el siguiente pipeline: Entrenamiento->Cuantizacion->Simulacion->Estudio de Errores en los buffer.
   - Errors: Carpeta con los errores (numericos) probados para cada red.
   - Stats: Carpeta con el duty de cada buffer (numerico) para cada red.
   - Trained Weights: Carpeta con los pesos entrenados de cada red (solo se incluyen cuando su peso es tolerado por github).
   - Figures: Carpeta de resultados para cada red,dataset,peso entrenado, incluye a:
     - Quantization Experiments: Contiene una imagen con los resultados de accuracy y/o Loss de la red (provado en todo el conjunto de datos de prueba) en base al numero de bits de las activaciones y pesos (independientemente).
     - Buffer Heatmaps: Contiene Imagenes de el duty del buffer considerando cada direccion de tamaño igual a las activaciones, ademas de un grafico de duty promedio segun posicion del bit.
     - Errores: Contiene Imagenes de la Accuracy y/o Loss para cada buffer, para cada seccion del buffer, para distinto numero de errores, ademas BoxPlots de estos.  Las secciones estan determinadas de acuerdo al numero de capas que las utilizan (por ejemplo la primera seccion suele ser utilizada por todas las capas)
    
### Particularidades:
   - PilotNet:
     - Dataset Basado en [Udacity's Self-Driving Car Nanodegree](https://github.com/udacity/self-driving-car-sim)
     - Se testearon 3953 imagenes durante la simulacion del duty
     - Cada error mostrado en la carpeta Errors es el promedio del error de 200 muestras para cada combinacion de: (tipo de error, buffer, seccion, numero de errors)
     - Se incluyo por separado una imagen de errores en la parte baja del buffer (primeras secciones), usando un numero absoluto de errores en lugar a uno relativo (usado en secciones de mayor tamaño)
