from numba import cuda
import numpy as np
from timeit import default_timer as timer
from tensorflow.python.keras import backend as K
import pickle


def save_obj(obj, obj_dir ):
	with open( obj_dir + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# Cargar objeto desde el subdirectorio obj
def load_obj(obj_dir):
	with open( obj_dir + '.pkl', 'rb') as f:
		return pickle.load(f)


Malpc = 12    #Maximo de activaciones leidas en un ciclo
Maepc = 12    #Maximo de activaciones escritas en un ciclo

#Cuda
threadsperblock = 512

#Parametros de la matriz de dsp slices.
filas      = Malpc
dsp_slices = 720
columnas   = dsp_slices/filas

#Parametros de Buffers de pesos
Mpclpc = 24   # Maximo de pesos convolucionales leidos en un ciclo
Mpflpc = 48   # Maximo de pesos full conectados leidos en un ciclo

@cuda.jit
def update_duty(data_buffer,stats_buffer,n_cycles):
	pos = cuda.grid(1)
	if pos < stats_buffer.size and data_buffer[pos] == 1:
		stats_buffer[pos] += n_cycles            

def intarr_to_binarr(intarr,word_size):
	binList = []
	for integer in intarr:
		if integer < 0:
			binList.append(np.binary_repr(-integer + 2**(word_size-1), width=word_size))
		else:
			binList.append(np.binary_repr(integer, width=word_size))
	return np.array(list(''.join(binList)),dtype=int)


def Get_datos_flujo_Conv(N_Filtros,Kernel_size,stride,N_inputs_channels,input_dim,Malpc,Maepc,Mpclpc,filas,columnas):
	Data = {}
	Data['latencia']                = int((Kernel_size**2)*N_inputs_channels)
	Data['ciclos_disponibles']      = int(Kernel_size*N_inputs_channels*stride)   
	GR                              = np.ceil(Kernel_size/stride)
	FU                              = min(Malpc, filas, input_dim)
	CU                              = min(Maepc*GR*Data['ciclos_disponibles']/FU, int(columnas/GR)*GR, N_Filtros*GR)
	Data['grado_paralelismo_usado'] = int(min(CU/GR,N_Filtros,Mpclpc))
	return Data

def Get_datos_flujo_FC(input_size,output_size,Malpc,Mpflpc,filas,columnas):
	Data = {}
	Data['filas_usadas']      = int(min(Malpc,filas, input_size))
	Data['columnas_usadas']   = int(min(np.floor(Mpflpc/Data['filas_usadas']),output_size,columnas))
	Data['latencia']          = int(np.ceil(input_size/Data['filas_usadas']))
	return Data


def generate_List(Rows,Columns,Filters,Maepc,Paralelismo,Latencia,Ciclos_requeridos,offset=0):
	Low_Index_List   = []
	High_Index_List  = []
	Cycle_List       = []
	for li in range(0,Filters,Paralelismo):
		for c  in range(0,Columns,Maepc):
			for r in range(0,Rows):
				tmp_list1 = []
				tmp_list2 = []
				for am in range(li,min(li+Paralelismo,Filters)):
					if r == 0 and am == li:
						Cycle_List.append(Latencia)
					elif am == li:
						Cycle_List.append(Ciclos_requeridos)
					tmp_list1.append(Rows*Columns*am + Columns*r + c + offset)    
					if c == Columns//Maepc*Maepc:
						tmp_list2.append(Rows*Columns*am + Columns*r + c + Columns%Maepc + offset)
					else:
						tmp_list2.append(Rows*Columns*am + Columns*r + c + Maepc + offset)
				Low_Index_List.append(tmp_list1)
				High_Index_List.append(tmp_list2)
	return Low_Index_List,High_Index_List,Cycle_List


def Sim_Conv(Input_dim, Output_dim_V ,Output_dim_H, Input_channels, Stride, Kernel_size, N_Filtros, data_buffer_p,
			 duty_p, data_buffer_s, duty_s, layer_outputs, word_size, frac_size, offset = 0):
	# Obtener Metadatos de flujo de datos
	Flow_Data = Get_datos_flujo_Conv(N_Filtros,Kernel_size,Stride,Input_channels,Input_dim,Malpc,Maepc,Mpclpc,filas,columnas)
	# Obtener indices de activaciones a escribir (en orden secuencial temporal) y el costo en ciclos.
	LIL,HIL,CL = generate_List(Output_dim_V,Output_dim_H,N_Filtros,Maepc,Flow_Data['grado_paralelismo_usado'],Flow_Data['latencia'],Flow_Data['ciclos_disponibles'])
	# Tamaño del trozo de buffer usado (en bits)
	limit = max(max(HIL))*word_size
	duty_tmp = np.zeros(duty_p[0:limit].size,dtype=np.uint16)
	blockspergrid = (duty_tmp.size + (threadsperblock - 1)) // threadsperblock
	contador = 0
	for cycles, low_idx, high_idx in zip(CL, LIL, HIL):
		contador += cycles
		# Actualizar el duty
		update_duty[blockspergrid, threadsperblock](data_buffer_p[0:limit],duty_tmp,cycles)
		# To evade overflow
		if contador > 60000:
			duty_p[offset*word_size:offset*word_size+limit] += duty_tmp
			duty_tmp[:] = 0
			contador = 0
		# Get and write new activation
		for lidx,hidx in zip(low_idx, high_idx):
			out = layer_outputs[lidx:hidx]
			bin_data = intarr_to_binarr(np.floor(out*2**frac_size).astype(np.int32),word_size)
			data_buffer_p[(offset+lidx)*word_size:(offset+lidx)*word_size + len(bin_data)] = bin_data
	# traspasar residuos
	duty_p[offset*word_size:offset*word_size+limit] += duty_tmp
	ciclos = np.sum(CL)
	# update the rest of the primary array
	if limit < duty_p.size:
		duty_p[offset*word_size+limit:][data_buffer_p[offset*word_size+limit:]==1] += ciclos
	if offset > 0:
		duty_p[0:offset*word_size][data_buffer_p[0:offset*word_size]==1] += ciclos
	# update the secundary array
	duty_s[data_buffer_s==1] += ciclos
	return ciclos


def Sim_DWConv(Input_dim, Output_dim_V ,Output_dim_H, Input_channels, Stride, Kernel_size, data_buffer_p,
			 duty_p, data_buffer_s, duty_s, layer_outputs, word_size, frac_size):
	# Get Dataflow Data
	ciclos = 0
	offset = 0
	for filt in range(Input_channels):
		Flow_Data = Get_datos_flujo_Conv(1,Kernel_size,Stride,1,Input_dim,Malpc,Maepc,Mpclpc,filas,columnas)
		# Get List of index, and cycles
		LIL,HIL,CL = generate_List(Output_dim_V,Output_dim_H,1,Maepc,Flow_Data['grado_paralelismo_usado'],Flow_Data['latencia'],Flow_Data['ciclos_disponibles'],offset)
		# Using a temporal array of only the size needed
		low_limit  = min(min(LIL))*word_size
		high_limit = max(max(HIL))*word_size
		duty_tmp = np.zeros(duty_p[low_limit:high_limit].size,dtype=np.uint8)
		blockspergrid = (duty_tmp.size + (threadsperblock - 1)) // threadsperblock
		contador = 0
		for cycles, low_idx, high_idx in zip(CL, LIL, HIL):
			contador += cycles
			# Update Duty
			update_duty[blockspergrid, threadsperblock](data_buffer_p[low_limit:high_limit],duty_tmp,cycles)
			# To evade overflow
			if contador > 200:
				duty_p[low_limit:high_limit] += duty_tmp
				duty_tmp[:] = 0
				contador = 0
			# Get and write new activation
			out = layer_outputs[low_idx[0]:high_idx[0]]
			bin_data = intarr_to_binarr(np.floor(out*2**frac_size).astype(np.int32),word_size)
			data_buffer_p[low_idx[0]*word_size:low_idx[0]*word_size + len(bin_data)] = bin_data
		# traspasar residuos
		duty_p[low_limit:high_limit] += duty_tmp
		ciclos += np.sum(CL)
		offset += Output_dim_V*Output_dim_H
		# update the rest of the primary array
		if high_limit < duty_p.size:
			duty_p[high_limit:][data_buffer_p[high_limit:]==1] += np.sum(CL)
		if low_limit > 0:
			duty_p[:low_limit][data_buffer_p[:low_limit]==1] += np.sum(CL)
	# update the secundary array
	duty_s[data_buffer_s==1] += ciclos
	return ciclos


def Sim_FC(Input_dim, Output_dim, layer_outputs, data_buffer_p, duty_p, data_buffer_s, duty_s, word_size, frac_size):
	# Obtencion de los metadatos del flujo de datos de la capa.
	Flow_Data = Get_datos_flujo_FC(Input_dim,Output_dim,Malpc,Mpflpc,filas,columnas)
	ciclos = 0
	#blockspergrid = (duty_p.size + (threadsperblock - 1)) // threadsperblock
	for packet in range(0,Output_dim,Flow_Data['columnas_usadas']):
		ciclos += Flow_Data['latencia']
		#update_duty[blockspergrid, threadsperblock](data_buffer_p,duty_p,Flow_Data['latencia'])
		duty_p[data_buffer_p==1] += Flow_Data['latencia']
		out = layer_outputs[0,packet:packet + Flow_Data['columnas_usadas']]
		bin_data = intarr_to_binarr(np.floor(out*2**frac_size).astype(np.int32),word_size)
		data_buffer_p[packet*word_size:packet*word_size + len(bin_data)] = bin_data
	duty_s[data_buffer_s==1] += ciclos
	return ciclos

def Sim_Input_and_Pooling(Activaciones_por_ciclo, N_act, layer_outputs, data_buffer_p, duty_p, data_buffer_s, duty_s, word_size, frac_size):
	ciclos = 0
	limit = N_act*word_size
	duty_tmp = np.zeros(duty_p[:limit].size,dtype=np.uint8)
	blockspergrid = (duty_tmp.size + (threadsperblock - 1)) // threadsperblock
	contador = 0
	for packet in range(0, N_act, Activaciones_por_ciclo):
		ciclos += 1
		contador += 1
		update_duty[blockspergrid, threadsperblock](data_buffer_p[0:limit],duty_tmp,1)
		if contador == 255:
			duty_p[:limit] += duty_tmp
			duty_tmp[:] = 0
			contador = 0
		out = layer_outputs[packet:packet + Activaciones_por_ciclo]
		bin_data = intarr_to_binarr(np.floor(out*2**frac_size).astype(np.int32),word_size)
		data_buffer_p[packet*word_size:packet*word_size + len(bin_data)] = bin_data
	duty_p[:limit] += duty_tmp
	if limit < duty_p.size:
		duty_p[limit:][data_buffer_p[limit:]==1] += ciclos
	# update the secundary array
	duty_s[data_buffer_s==1] += ciclos
	return ciclos


def get_all_outputs(model, input_data, learning_phase=False):
	outputs = [layer.output for layer in model.layers] # exclude Input
	layers_fn = K.function([model.input], outputs)
	return layers_fn([input_data])



def AlexNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 154587, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=227, Output_dim_V=55, Output_dim_H = 55, Input_channels=3, Stride=4, Kernel_size=11, N_Filtros=96, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 69984, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=31, Output_dim_V=27, Output_dim_H = 27, Input_channels=96, Stride=1, Kernel_size=5, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 43264, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=13, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=13, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 9216, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=9216, Output_dim=4096, layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=Classes, layer_outputs= activaciones[11], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	return ciclos

def VGG16_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	ciclos = 0
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=226, Output_dim_V=224, Output_dim_H = 224, Input_channels=3, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=226, Output_dim_V=224, Output_dim_H = 224, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 802816, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2, 
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=114, Output_dim_V=112, Output_dim_H = 112, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=128, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=114, Output_dim_V=112, Output_dim_H = 112, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=128, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 401408, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704, layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352, layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352, layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=25088, Output_dim=4096, layer_outputs= activaciones[19], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[20], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=Classes, layer_outputs= activaciones[21], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	return ciclos

def PilotNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 39600, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=200, Output_dim_V=98, Output_dim_H = 31, Input_channels=3, Stride=2, Kernel_size=5, N_Filtros=24, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=98, Output_dim_V=47, Output_dim_H = 14, Input_channels=24, Stride=2, Kernel_size=5, N_Filtros=36, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=47, Output_dim_V=22, Output_dim_H = 5, Input_channels=36, Stride=2, Kernel_size=5, N_Filtros=48, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=22, Output_dim_V=20, Output_dim_H = 3, Input_channels=48, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=20, Output_dim_V=18, Output_dim_H = 1, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=1152, Output_dim=1164, layer_outputs= activaciones[6], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=1164, Output_dim=100, layer_outputs= activaciones[7], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=100, Output_dim=50,layer_outputs= activaciones[8], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=50, Output_dim=10,layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=10, Output_dim=1,layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	return ciclos


def MobileNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=227, Output_dim_V=113, Output_dim_H = 113, Input_channels=3, Stride=2, Kernel_size=3, N_Filtros=32,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=115, Output_dim_V=113 ,Output_dim_H=113, Input_channels=32, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=113, Output_dim_V=113, Output_dim_H = 113, Input_channels=32, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=114, Output_dim_V=56 ,Output_dim_H=56, Input_channels=64, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=64, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=58, Output_dim_V=56 ,Output_dim_H=56, Input_channels=128, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=57, Output_dim_V=28 ,Output_dim_H=28, Input_channels=128, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=30, Output_dim_V=28 ,Output_dim_H=28, Input_channels=256, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=29, Output_dim_V=14 ,Output_dim_H=14, Input_channels=256, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)   
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[19],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[20],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[21],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[22],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[23],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=15, Output_dim_V=7 ,Output_dim_H=7, Input_channels=512, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[24],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=1024,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[25],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=8, Output_dim_V=7 ,Output_dim_H=7, Input_channels=1024, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[26],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=1024, Stride=1, Kernel_size=1, N_Filtros=1024,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[27],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Global Average Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 1024, layer_outputs = np.swapaxes(np.swapaxes(activaciones[28],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=1, Output_dim_V=1, Output_dim_H = 1, Input_channels=1024, Stride=1, Kernel_size=1, N_Filtros=Classes,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[29],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	return ciclos

def ZFNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=224, Output_dim_V=109, Output_dim_H = 109, Input_channels=3, Stride=2, Kernel_size=7, N_Filtros=96, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 279936, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=57, Output_dim_V=27, Output_dim_H = 27, Input_channels=96, Stride=2, Kernel_size=5, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 43264, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 9216, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=9216, Output_dim=4096, layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=Classes, layer_outputs= activaciones[11], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	return ciclos

def SqueezeNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=229, Output_dim_V=112, Output_dim_H = 112, Input_channels=3, Stride=2, Kernel_size=7, N_Filtros=96,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 301056,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Fire 1")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=96, Stride=1, Kernel_size=1, N_Filtros=16,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=16, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=16, Stride=1, Kernel_size=3, N_Filtros=64, offset = 200704,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Fire 2")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=16,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=16, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=16, Stride=1, Kernel_size=3, N_Filtros=64, offset = 200704,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Fire 3")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=32,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=32, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=32, Stride=1, Kernel_size=3, N_Filtros=128, offset = 401408,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Fire 4")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=32,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=32, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=32, Stride=1, Kernel_size=3, N_Filtros=128, offset = 100352,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Fire 5")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=48,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=48, Stride=1, Kernel_size=1, N_Filtros=192,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=48, Stride=1, Kernel_size=3, N_Filtros=192, offset = 150528,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Fire 6")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=48,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[19],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=48, Stride=1, Kernel_size=1, N_Filtros=192,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[20],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=48, Stride=1, Kernel_size=3, N_Filtros=192, offset = 150528,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[21],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Fire 7")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[22],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=64, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[23],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=256, offset = 200704,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[24],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa MaxPooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[25],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Fire 8")
	#print("Capa Squeeze")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[26],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Expand1x1")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=64, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[27],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Expand3x3")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=256, offset = 50176,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[28],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Conv")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=Classes, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[29],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Global Average Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = Classes, layer_outputs = activaciones[30].flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	return ciclos



def DenseNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size,Classes):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=224, Output_dim_V=112, Output_dim_H = 112, Input_channels=3, Stride=2, Kernel_size=7, N_Filtros=64, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Dense 1")
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=64, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 200704,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 301056,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=96, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 301056,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 401408,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 401408,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 501760,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=160, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 501760,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 602112,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=192, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 602112,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 702464,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=224, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 702464,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[19],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 802816,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[20],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	# Transicion 1
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[21],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[22],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Dense 2")
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[23],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 100352,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[24],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 125440,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[25],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=160, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[26],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 125440,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[27],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[28],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=192, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[29],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 150528,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[30],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 175616,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[31],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=224, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[32],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 175616,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[33],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[34],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[35],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 200704,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[36],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 225792,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[37],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=288, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[38],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 225792,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[39],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 250880,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[40],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=320, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[41],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 250880,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[42],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 275968,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[43],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=352, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[44],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 275968,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[45],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 301056,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[46],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[47],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 301056,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[48],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 326144,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[49],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=416, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[50],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 326144,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[51],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 351232,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[52],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=448, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[53],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 351232,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[54],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 376320,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[55],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=480, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[56],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 376320,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[57],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 401408,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[58],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	# Transicion 2
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[59],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 50176,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[60],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Dense 3")
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[61],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 50176,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[62],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 56448,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[63],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=288, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[64],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 56448,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[65],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 62720,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[66],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=320, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[67],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 62720,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[68],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 68992,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[69],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=352, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[70],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 68992,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[71],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 75264,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[72],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[73],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 75264,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[74],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 81536,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[75],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=416, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[76],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 81536,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[77],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 87808,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[78],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=448, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[79],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 87808,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[80],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 94080,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[81],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=480, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[82],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 94080,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[83],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[84],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[85],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 100352,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[86],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 106624,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[87],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=544, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[88],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 106624,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[89],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 112896,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[90],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=576, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[91],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 112896,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[92],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 119168,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[93],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=608, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[94],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 119168,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[95],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 125440,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[96],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=640, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[97],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 125440,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[98],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 131712,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[99],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=672, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[100],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 131712,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[101],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 137984,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[102],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=704, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[103],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 137984,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[104],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 144256,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[105],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=736, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[106],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 144256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[107],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[108],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=768, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[109],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 150528,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[110],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 156800,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[111],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=800, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[112],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 156800,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[113],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 163072,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[114],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=832, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[115],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 163072,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[116],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 169344,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[117],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=864, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[118],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 169344,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[119],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 175616,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[120],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=896, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[121],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 175616,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[122],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 181888,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[123],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=928, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[124],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 181888,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[125],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 188160,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[126],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=960, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[127],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 188160,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[128],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 194432,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[129],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=992, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[130],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 194432,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[131],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[132],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	# Transicion 3
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=1024, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[133],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 25088,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[134],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Dense 4")
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[135],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 25088,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[136],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 26656,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[137],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=544, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[138],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 26656,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[139],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 28244,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[140],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=576, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[141],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 28244,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[142],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 29792,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[143],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=608, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[144],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 29792,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[145],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 31360,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[146],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=640, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[147],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 31360,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[148],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 32928,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[149],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=672, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[150],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 32928,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[151],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 34496,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[152],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=704, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[153],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 34496,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[154],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 36064,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[155],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=736, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[156],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 36064,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[157],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 37632,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[158],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=768, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[159],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 37632,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[160],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 39200,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[161],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=800, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[162],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 39200,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[163],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 40768,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[164],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=832, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[165],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 40768,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[166],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 42336,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[167],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=864, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[168],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 42336,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[169],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 43904,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[170],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=896, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[171],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 43904,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[172],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 45472,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[173],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=928, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[174],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 45472,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[175],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 47040,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[176],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=960, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[177],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 47040,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[178],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 48608,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[179],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_2, duty_p = duty_2, 
									data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=992, Stride=1, Kernel_size=1, N_Filtros=128, 
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[180],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=9, Output_dim_V=7, Output_dim_H = 7, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=32, offset = 48608,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[181],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("BatchNormalization")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 50176,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[182],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print("Global Average Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 1024, layer_outputs = activaciones[183].flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2, 
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=1024, Output_dim=Classes, layer_outputs= activaciones[184], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)

	return ciclos
