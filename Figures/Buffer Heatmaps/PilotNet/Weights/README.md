# Imagenes obtenidas tras la simulacion del buffer.

### Archivos: 
   - Buffer 1 Higher Portion: Muestra la porción del buffer con duty > 0.7, es decir celdas que tienen al menos un 70% del tiempo almacenado un valor 1 (En este caso no hay).
   - Duty Buffer 1: Muestra el duty del buffer, valores en torno a 0.5 es ideal. se muestran los resultados en un rango de 0 a 0.5, asi por ejemplo un valor de 0.1 se muestra igual a un 0.9 pues ambos tienen el mismo efecto en el desgaste del buffer.
   - Duty per bit position: Duty promedio segun la posición del bit.

### Parametros de simulación:
   - Numero de Macs: 720
   - Activaciones Leidas/Escritas por ciclo: 12
   - Duty per bit position: Duty promedio segun la posición del bit.
   - Activaciones: 1 bit signo + 7 bits parte fraccionaria
   - Pesos: 1 bit signo + 7 bits parte fraccionaria
   - Numero de Imagenes simuladas: 3953
