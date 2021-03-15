import numpy as np
from functions import write_conv_output, buffer_stadistics, buffer_stadistics_opt, get_all_outputs
import tensorflow as tf

# Metadatos para las capas simuladas
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#     Latencia: ciclos requeridos para obtener la primera salida desde la primera escritura de datos
#     ciclos_disponibles: Numero de ciclos entre la obtencion de un nuevo "paquete" de salidas disponibles para escribir en memoria.
#     grado_reusabilidad: Numero de "paquetes" de salidas procesandose con "desfase" por cada "paquete" en paralelo
#     grado_paralelismo:  Numero de "paquetes" de salidas procesandose en paralelo
#     filas_usadas:       Numero de filas usadas del total disponible en la matriz de DSP_SLICES
#     columnas_usadas:    Numero de columnas usadas del total disponible en la matriz de DSP_SLICES
#     throughput:         Numero de salidas procesadas por ciclo (en promedio)
#     ciclos_requeridos:  Numero de ciclos requeridos para escribir en memoria cada grupo de paquetes obtenido del paralelismo.
#     ciclos_teoricos:    Ciclos requeridos en teoria para el procesamiento de la capa asumiendo 100% de utilizacion de la escritura y las Macs, sin restriccion de lectura
#     ciclos_hardware:    Ciclos requeridos en la practica para el procesamiento de la capa bajo la arquitectura propuesta
#     ciclos_simulacion:  Ciclos requeridos para la ejecucion de la simulacion de la capa(para estimacion del tiempo de simulacion)

def Get_datos_flujo_Conv(N_Filtros,Kernel_size,stride,N_inputs_channels,input_dim,output_dim,Malpc,Maepc,Mpclpc,filas,columnas):
    Data = {}
    Data['latencia']                = (Kernel_size**2)*N_inputs_channels
    Data['ciclos_disponibles']      = Kernel_size*N_inputs_channels*stride    
    Data['grado_reusabilidad']      = np.ceil(Kernel_size/stride)
    Data['filas_usadas']            = min(Malpc, filas, input_dim)
    Data['columnas_usadas']         = min(Maepc*Data['grado_reusabilidad']*Data['ciclos_disponibles']/Data['filas_usadas'], int(columnas/Data['grado_reusabilidad'])*Data['grado_reusabilidad'], N_Filtros*Data['grado_reusabilidad'])
    Data['grado_paralelismo_usado'] = min(Data['columnas_usadas']/Data['grado_reusabilidad'],N_Filtros,Mpclpc)
    Data['throughput_usado']        = Data['grado_paralelismo_usado']*Data['filas_usadas']/Data['ciclos_disponibles']
    Data['ciclos_requeridos']       = Data['grado_paralelismo_usado']*Data['filas_usadas']/Maepc
    Multiplicaciones_requeridas     = output_dim**2*N_Filtros*Kernel_size**2*N_inputs_channels
    Total_de_Macs                   = filas*columnas
    N_escrituras                    = output_dim**2*N_Filtros
    Data['ciclos_teoricos']         = np.ceil(max(Multiplicaciones_requeridas/Total_de_Macs, N_escrituras/Maepc))
    Data['ciclos_hardware']         = (Data['latencia']+(output_dim-1)*Data['ciclos_disponibles'])*np.ceil(output_dim/Data['filas_usadas'])*np.ceil(N_Filtros/Data['grado_paralelismo_usado'])
    Data['ciclos_simulacion']       = output_dim*np.ceil(output_dim/Data['filas_usadas'])*Data['grado_paralelismo_usado']*np.ceil(N_Filtros/Data['grado_paralelismo_usado'])
    #Data['grado_paralelismo_disponible']     =  min(np.floor(columnas/Data['grado_reusabilidad']),N_Filtros)
    #Data['throughput_disponible'] =  Data['grado_paralelismo_disponible']*Data['filas_usadas']/Data['ciclos_disponibles']
    return Data

def Get_datos_flujo_FC(input_size,output_size,Malpc,Maepc,Mpflpc,filas,columnas):
    Data = {}
    Data['filas_usadas']      = min(Malpc,filas, input_size)
    Data['columnas_usadas']   = min(np.floor(Mpflpc/Data['filas_usadas']),output_size,columnas)
    Data['latencia']          = np.ceil(input_size/Data['filas_usadas'])
    Data['throughput']        = Data['columnas_usadas']/Data['latencia']
    Data['ciclos_disponibles'] = Data['latencia']
    Data['ciclos_requeridos']  = np.ceil(Data['columnas_usadas']/Maepc)
    # Se asume que se pueden escribir las salidas sin "stall"
    return Data


# Simulacion de capas
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# data_buffer_p / stats_buffer_p: refierase a el buffer primario aquel usado para escribir las salidas
# data_buffer_s / stats_buffer_s: refierase a el buffer secundario aquel usado para leer las entradas
# Word_bits: bits de cuantizacion de las activaciones
# Malpc:    Maximo numero de activaciones leidas en un ciclo
# Maepc:    Maximo numero de activaciones escritas en un ciclo
# Mpclpc:   Maximo numero de pesos convolucionales leidos en un ciclo
# filas:    Numero de filas de la matriz de DSP_slices
# columnas: Numero de columnas de la matriz de DSP_slices
# Simulacion de capa Convolucional
def Sim_Conv(Input_dim, Output_dim, Input_channels, Stride, Kernel_size, N_Filtros, data_buffer_p,
             stats_buffer_p, data_buffer_s, stats_buffer_s, layer_outputs):
    # Offsets para escritura en memoria
    map_offset    = Output_dim**2*Word_bits
    row_offset    = Output_dim*Word_bits
    column_offset = Word_bits
    # Metadatos para el flujo de los datos
    Flow_Data = Get_datos_flujo_Conv(N_Filtros,Kernel_size,Stride,Input_channels,Input_dim,Output_dim,Malpc,Maepc,Mpclpc,filas,columnas)
    # Indices de lectura/escritura
    row = column = act_map = base_map = 0
    contador = 0
    ciclos   = 0
    while True:
        # Las convoluciones de la primera fila no aprovechan el ingreso de datos de convoluciones previas
        if row == 0 and act_map == base_map:
            delta  = Flow_Data['latencia']
            contador += 1
        # Se ha terminado de escribir el ultimo subpaquete de bits, no hay cambios en la memoria hasta que se tenga un nuevo paquete
        elif act_map == base_map:
            delta  = Flow_Data['ciclos_disponibles'] - contador
            contador = 0
        # Se escribe un subpaquete de bits en uno de los feature maps.
        else:
            delta = 1
            contador += 1
        ciclos = ciclos + int(delta)
        # Obtencion del subpaquete a escribir en 1 feature map. 
        out = layer_outputs[0,row,column:column+Flow_Data['filas_usadas'],act_map]
        # Escritura del paquete
        offset = act_map*map_offset + row*row_offset + column*column_offset
        write_conv_output(out, offset, data_buffer_p, Word_bits = Word_bits, Frac_bits = Frac_bits)
        # Se actualiza la estadisticas de los buffers
        buffer_stadistics_opt(out, offset,stats_buffer_p,int(delta))
        # Logica de siguiente paquete.
        act_map = (act_map + 1) % Flow_Data['grado_paralelismo_usado'] + base_map
        act_map = int(act_map)
        if act_map == base_map or act_map > N_Filtros-1:
            act_map = base_map
            row = row + 1
        if row == Output_dim:
            row = 0
            column = column + Flow_Data['filas_usadas']
        if column >= Output_dim:
            column = 0
            base_map = base_map + Flow_Data['grado_paralelismo_usado']
            base_map = int(base_map)
            act_map = base_map
        if base_map > N_Filtros - 1:
            break
    buffer_stadistics(data_buffer_s,stats_buffer_s,ciclos)
    return ciclos

# Simulacion de capa MaxPooling / Average Pooling / Batch Normalization (por el momento abstracto)
def Sim_MP_BN(salidas_por_ciclo, Input_channels, Output_dim, data_buffer_p, stats_buffer_p,
	      data_buffer_s, stats_buffer_s, layer_outputs):
    # Offsets para escritura en memoria
    map_offset    = Output_dim**2*Word_bits
    row_offset    = Output_dim*Word_bits
    column_offset = Word_bits
    # Metadatos para el flujo de los datos
    # Indices de lectura/escritura
    row = column = act_map = base_map = 0
    contador = 0
    ciclos   = 0
    while True:
        ciclos += 1
        # Obtencion del subpaquete a escribir en 1 feature map. 
        out = layer_outputs[0,row,column:column+salidas_por_ciclo,act_map]
        # Escritura del paquete
        offset = act_map*map_offset + row*row_offset + column*column_offset
        write_conv_output(out, offset, data_buffer_p, Word_bits = Word_bits, Frac_bits = Frac_bits)
        # Se actualiza la estadisticas de los buffers
        buffer_stadistics_opt(out, offset,stats_buffer_p,1)
        # Logica de siguiente paquete.
        act_map += 1
        if act_map == Input_channels:
            act_map = 0
            row = row + 1
        if row == Output_dim:
            row = 0
            column = column + salidas_por_ciclo
        if column >= Output_dim:
            break
    buffer_stadistics(data_buffer_s,stats_buffer_s,ciclos)
    return ciclos

# Simulacion de capa Full Conectada
def Sim_FC(Input_dim, Output_dim, layer_outputs, data_buffer_p, stats_buffer_p, 
 	   data_buffer_s, stats_buffer_s):
    # Obtencion de los metadatos del flujo de datos de la capa.
    Flow_Data = Get_datos_flujo_FC(Input_dim,Output_dim,Malpc,Maepc,Mpflpc,filas,columnas)
    # ciclos hasta que se obtiene el primer conjunto de salidas.
    # contador de salidas escritas en el buffer
    salidas_guardadas  = 0
    contador = 0
    ciclos = 0
    while True:
        if salidas_guardadas % Flow_Data['columnas_usadas'] == 0:
            delta = Flow_Data['ciclos_disponibles'] - contador
            contador = 0
        else:
            delta = 1
            contador += 1
        ciclos = ciclos + int(delta)
        out = layer_outputs[0,salidas_guardadas:salidas_guardadas + Maepc]
        write_conv_output(out, salidas_guardadas*Word_bits, data_buffer_p, Word_bits = Word_bits, Frac_bits = Frac_bits)
        buffer_stadistics_opt(out, salidas_guardadas,stats_buffer_p,int(delta))
        # Actualizacion de las salidas escritas en el buffer
        salidas_guardadas = salidas_guardadas + len(out)
        if salidas_guardadas == Output_dim:
            break
    buffer_stadistics(data_buffer_s, stats_buffer_s,ciclos)
    return ciclos

# Simulacion de redes
#------------------------------------------------------------------------------------------------------------------------
# Lenet
def Lenet_Simulation(imagen):
    #print("inicia simulacion")
    outputs = get_all_outputs(QLenet, tf.expand_dims(imagen,axis=0))
    ciclos = 0           
    #print("Capa Convolucional")
    # Datos de la capa
    ciclos += Sim_Conv(Input_dim=32, Output_dim=28, Input_channels=1, Stride=1, Kernel_size=5, N_Filtros=6,
                       data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                       data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                       layer_outputs = outputs[5])
    #print("Capa AvgPool")
    ciclos += Sim_MP_BN(Output_dim=14, Input_channels=6, salidas_por_ciclo= Maepc,
                        data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                        data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                        layer_outputs= outputs[6])
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=14, Output_dim=10, Input_channels=6, Stride=1, Kernel_size=5, N_Filtros=16,
                       data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                       data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                       layer_outputs = outputs[10])
    #print("Capa AvgPool")
    ciclos += Sim_MP_BN(Output_dim=5, Input_channels=16, salidas_por_ciclo= Maepc,
                        data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                        data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                        layer_outputs= outputs[11])   
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=400, Output_dim=120,
                     data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2,
                     data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                     layer_outputs= outputs[16])
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=120, Output_dim=84,
                     data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1,
                     data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                     layer_outputs= outputs[20])    
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=84, Output_dim=10,
                     data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2,
                     data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                     layer_outputs= outputs[24])    
    return ciclos

# AlexNet
def AlexNet_Simulation(imagen):
    #print("inicia simulacion")
    outputs = get_all_outputs(QAlexNet, tf.expand_dims(imagen,axis=0))
    ciclos = 0
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=227, Output_dim=55, Input_channels=3, Stride=4, Kernel_size=11, N_Filtros=96,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[5])
    #print("Capa BN")
    ciclos += Sim_MP_BN(Output_dim=55, Input_channels=96, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[7])
    #print("Capa MP")
    ciclos += Sim_MP_BN(Output_dim=27, Input_channels=96, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[8])
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=31, Output_dim=27, Input_channels=96, Stride=1, Kernel_size=5, N_Filtros=256,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[12])
    #print("Capa BN")
    ciclos += Sim_MP_BN(Output_dim=27, Input_channels=256, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[14])
    #print("Capa MP")
    ciclos += Sim_MP_BN(Output_dim=13, Input_channels=256, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[15])
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=15, Output_dim=13, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=384,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[19])
    #print("Capa BN")
    ciclos += Sim_MP_BN(Output_dim=13, Input_channels=384, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[21])
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=13, Output_dim=13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=384,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[25])
    #print("Capa BN")
    ciclos += Sim_MP_BN(Output_dim=13, Input_channels=384, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[27])
    #print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=13, Output_dim=13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=256,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[31])
    #print("Capa BN")
    ciclos += Sim_MP_BN(Output_dim=13, Input_channels=256, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[33])
    #print("Capa MP")
    ciclos += Sim_MP_BN(Output_dim=6, Input_channels=256, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[34])
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=9216, Output_dim=4096,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1,
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[39])
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=4096, Output_dim=4096,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2,
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[44])
    #print("Capa FC")
    ciclos += Sim_FC(Input_dim=4096, Output_dim=10,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1,
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[47])
    return ciclos

# VGG16
def VGG16_Simulation(imagen):
    print("inicia simulacion")
    t_trascurrido = time.time()
    outputs = get_all_outputs(QVGG16, tf.expand_dims(imagen,axis=0))
    print("Capa Convolucional")
    ciclos = 0
    ciclos += Sim_Conv(Input_dim=226, Output_dim=224, Input_channels=3, Stride=1, Kernel_size=3, N_Filtros=64,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[5])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=226, Output_dim=224, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=64,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[9])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa MaxPooling")
    ciclos += Sim_MP_BN(Output_dim=112, Input_channels=64, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[10])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=114, Output_dim=112, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=128,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[14])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=114, Output_dim=112, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=128,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[18])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa MaxPooling")
    ciclos += Sim_MP_BN(Output_dim=56, Input_channels=128, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[19])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=58, Output_dim=56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=256,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[23])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=58, Output_dim=56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[27])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=58, Output_dim=56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[31])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa MaxPooling")
    ciclos += Sim_MP_BN(Output_dim=28, Input_channels=256, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[32])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=30, Output_dim=28, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[36])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=30, Output_dim=28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[40])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=30, Output_dim=28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[44])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa MaxPooling")
    ciclos += Sim_MP_BN(Output_dim=14, Input_channels=512, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[45])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=16, Output_dim=14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[49])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=16, Output_dim=14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                      data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                      layer_outputs = outputs[53])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Convolucional")
    ciclos += Sim_Conv(Input_dim=16, Output_dim=14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512,
                      data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2, 
                      data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                      layer_outputs = outputs[57])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa MaxPooling")
    ciclos += Sim_MP_BN(Output_dim=7, Input_channels=512, salidas_por_ciclo= Maepc,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1, 
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[58])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Full Conectada")
    ciclos += Sim_FC(Input_dim=25088, Output_dim=4096,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2,
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[63])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Full Conectada")
    ciclos += Sim_FC(Input_dim=4096, Output_dim=4096,
                    data_buffer_p = IOBuffer_1, stats_buffer_p = stats_IOBuffer_1,
                    data_buffer_s = IOBuffer_2, stats_buffer_s = stats_IOBuffer_2,
                    layer_outputs= outputs[67])
    print('tiempo transcurrido:',time.time()-t_trascurrido,'ciclos:',ciclos)
    print("Capa Full Conectada")
    ciclos += Sim_FC(Input_dim=4096, Output_dim=10,
                    data_buffer_p = IOBuffer_2, stats_buffer_p = stats_IOBuffer_2,
                    data_buffer_s = IOBuffer_1, stats_buffer_s = stats_IOBuffer_1,
                    layer_outputs= outputs[71])
    return ciclos


