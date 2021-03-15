# Deep-Learning-Proyect
Repository for the proyect of the Digital Design of Embedded Systems course

El proyecto esta desarrollado en base a tensorflow 2.1 y keras.

Por el momento se ha dividido el trabajo en software en 3 las siguientes carpetas: Lenet5, AlexNet y VGG16. Dentro de cada carpeta existen 4 archivos: 

Training: Destinado al entrenamiento de la red, usada solamente para obtener los pesos que se cargaran al modelo.

Analysis: En este modelo se cargan los pesos entrenados en la red original y en una version cuantizada de esta, luego se comparan las activaciones de cada capa de la red original con la de su version cuantizada. Finalmente se realiza un analisis de los valores maximos y minimos de las salidas de cada capa y los pesos del modelo.

Quantization Test: Se compara la accuracy del modelo original con versiones quantizadas de este variando el numero de bits destinadas a la parte fraccional y/o entera.

Simulacion Buffers: procesa una simulacion de inferencia de datos en hardware, colecta estadisticas de los buffers.

Los notebooks creados para las redes comparten 3 archivos de funciones:

functions.py: Contiene funciones de proposito general
models.py: Contiene la definicion (arquitectura) de los modelos y su cuantizacion
simulation.py: Contiene la definicion de la simulacion de las redes y las capas en base al hardware.

Finalmente en la carpeta modelo hardware simulado se muestra un esquema con el flujo de datos (ciclos de escritura, ciclos de lectura y uso de las Macs)

El archivo training genera la carpeta TrainedWeights con los pesos entrenados (para VGG16 y AlexNet no se subio de antemano los pesos ya entrenados debido al tamaño del archivo.)


