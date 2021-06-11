# Imagenes obtenidas tras la inferencia con errores en buffer.

### Archivos: 
   - Se presentan dos experimentos, un numero de errores relativo (usado para secciones del buffer de tamaño considerable) y un numero de errores absoluto (usado en las secciones del buffer de menor tamaño)
   - En el caso del primer experimento se presenta ademas la imagen sin escalar para notar el aumento de la funcion Loss en funcion del numero de errores y la imagen escalada, para comparar este efecto seccion a seccion.
   - Por ultimo para cada experimento se muestran los boxplots obtenidos.

### Parametros de simulación:
   - Numero de muestras: 200
   - Numero de Imagenes simuladas: 3953 (todo el dataset destinado inferencia)
   - Tipos de error: valor estatico '1' en el primer o segundo bit.

### Cuantización:
   - Activaciones: 1 bit signo + 7 bits parte fraccionaria
   - Pesos: 1 bit signo + 7 bits parte fraccionaria
