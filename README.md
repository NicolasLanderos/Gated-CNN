# Deep-Learning-Proyect
Repository for the proyect of the Digital Design of Embedded Systems course

El proyecto esta desarrollado en base a tensorflow 2.1 y keras.

Por el momento se ha dividido el trabajo en software en 3 las siguientes carpetas: Lenet5, AlexNet y VGG16. Dentro de cada carpeta existen 4 archivos: 
Training: Destinado al entrenamiento de la red, usada solamente para obtener los pesos que se cargaran al modelo.
Analysis: En este modelo se cargan los pesos entrenados en la red original y en una version cuantizada de esta, luego se comparan las activaciones de cada capa de la red original con la de su version cuantizada. Finalmente se realiza un analisis de los valores maximos y minimos de las salidas de cada capa y los pesos del modelo.
Quantization Test: Se compara la accuracy del modelo original con versiones quantizadas de este variando el numero de bits destinadas a la parte fraccional.
Dataflow: Contiene los comandos para ver el flujo de datos del modelo en tensorboard.

El archivo training genera dos carpetas: GraphData con los datos necesarios para la visualizacion del modelo en tensorboard y TrainedWeights con los pesos entrenados (para VGG16 y AlexNet no se subio de antemano los pesos ya entrenados debido al tamaño del archivo.)

