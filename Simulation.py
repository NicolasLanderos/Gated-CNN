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
			 duty_p, data_buffer_s, duty_s, layer_outputs, word_size, frac_size):
	# Get Dataflow Data
	Flow_Data = Get_datos_flujo_Conv(N_Filtros,Kernel_size,Stride,Input_channels,Input_dim,Malpc,Maepc,Mpclpc,filas,columnas)
	# Get List of index, and cycles
	LIL,HIL,CL = generate_List(Output_dim_V,Output_dim_H,N_Filtros,Maepc,Flow_Data['grado_paralelismo_usado'],Flow_Data['latencia'],Flow_Data['ciclos_disponibles'])
	# Using a temporal array of only the size needed
	limit = max(max(HIL))*word_size
	duty_tmp = np.zeros(duty_p[:limit].size,dtype=np.uint16)
	blockspergrid = (duty_tmp.size + (threadsperblock - 1)) // threadsperblock
	contador = 0
	for cycles, low_idx, high_idx in zip(CL, LIL, HIL):
		contador += cycles
		# Update Duty
		update_duty[blockspergrid, threadsperblock](data_buffer_p[0:limit],duty_tmp,cycles)
		# To evade overflow
		if contador > 60000:
			duty_p[:limit] += duty_tmp
			duty_tmp[:] = 0
			contador = 0
		# Get and write new activation
		for lidx,hidx in zip(low_idx, high_idx):
			out = layer_outputs[lidx:hidx]
			bin_data = intarr_to_binarr(np.floor(out*2**frac_size).astype(np.int32),word_size)
			data_buffer_p[lidx*word_size:lidx*word_size + len(bin_data)] = bin_data
	# traspasar residuos
	duty_p[:limit] += duty_tmp
	ciclos = np.sum(CL)
	# update the rest of the primary array
	if limit < duty_p.size:
		duty_p[limit:][data_buffer_p[limit:]==1] += ciclos
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



def AlexNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 154587, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=227, Output_dim_V=55, Output_dim_H = 55, Input_channels=3, Stride=4, Kernel_size=11, N_Filtros=96, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 69984, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=31, Output_dim_V=27, Output_dim_H = 27, Input_channels=96, Stride=1, Kernel_size=5, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 43264, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=13, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=13, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=1, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 9216, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=9216, Output_dim=4096, layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=10,layer_outputs= activaciones[11], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	return ciclos

def VGG16_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	ciclos = 0
	#print("inicia simulacion")
	#print("Input")
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=226, Output_dim_V=224, Output_dim_H = 224, Input_channels=3, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=226, Output_dim_V=224, Output_dim_H = 224, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 802816, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2, 
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=114, Output_dim_V=112, Output_dim_H = 112, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=128, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=114, Output_dim_V=112, Output_dim_H = 112, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=128, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 401408, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=58, Output_dim_V=56, Output_dim_H = 56, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 200704, layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=30, Output_dim_V=28, Output_dim_H = 28, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352, layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=16, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=3, N_Filtros=512, layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 100352, layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=25088, Output_dim=4096, layer_outputs= activaciones[19], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[20], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=10,layer_outputs= activaciones[21], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	return ciclos

def PilotNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 39600, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=200, Output_dim_V=98, Output_dim_H = 31, Input_channels=3, Stride=2, Kernel_size=5, N_Filtros=24, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=98, Output_dim_V=47, Output_dim_H = 14, Input_channels=24, Stride=2, Kernel_size=5, N_Filtros=36, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=47, Output_dim_V=22, Output_dim_H = 5, Input_channels=36, Stride=2, Kernel_size=5, N_Filtros=48, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=22, Output_dim_V=20, Output_dim_H = 3, Input_channels=48, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=20, Output_dim_V=18, Output_dim_H = 1, Input_channels=64, Stride=1, Kernel_size=3, N_Filtros=64, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=1152, Output_dim=1164, layer_outputs= activaciones[6], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=1164, Output_dim=100, layer_outputs= activaciones[7], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=100, Output_dim=50,layer_outputs= activaciones[8], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=50, Output_dim=10,layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=10, Output_dim=1,layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	return ciclos


def MobileNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528,
									layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
									data_buffer_p = data_buffer_1, duty_p = duty_1, 
									data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=227, Output_dim_V=113, Output_dim_H = 113, Input_channels=3, Stride=2, Kernel_size=3, N_Filtros=32,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=115, Output_dim_V=113 ,Output_dim_H=113, Input_channels=32, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=113, Output_dim_V=113, Output_dim_H = 113, Input_channels=32, Stride=1, Kernel_size=1, N_Filtros=64,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=114, Output_dim_V=56 ,Output_dim_H=56, Input_channels=64, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=64, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=58, Output_dim_V=56 ,Output_dim_H=56, Input_channels=128, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=56, Output_dim_V=56, Output_dim_H = 56, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=128,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=57, Output_dim_V=28 ,Output_dim_H=28, Input_channels=128, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=128, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[9],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=30, Output_dim_V=28 ,Output_dim_H=28, Input_channels=256, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[10],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=28, Output_dim_V=28, Output_dim_H = 28, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=256,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[11],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=29, Output_dim_V=14 ,Output_dim_H=14, Input_channels=256, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[12],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=256, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[13],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[14],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[15],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[16],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[17],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)   
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[18],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[19],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[20],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[21],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=16, Output_dim_V=14 ,Output_dim_H=14, Input_channels=512, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[22],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=14, Output_dim_V=14, Output_dim_H = 14, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=512,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[23],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=15, Output_dim_V=7 ,Output_dim_H=7, Input_channels=512, Stride=2, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[24],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=512, Stride=1, Kernel_size=1, N_Filtros=1024,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[25],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa DWConvolucional")
	ciclos += Sim_DWConv(Input_dim=8, Output_dim_V=7 ,Output_dim_H=7, Input_channels=1024, Stride=1, Kernel_size=3,
						 layer_outputs = np.swapaxes(np.swapaxes(activaciones[26],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
						 data_buffer_p = data_buffer_1, duty_p = duty_1, 
						 data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=7, Output_dim_V=7, Output_dim_H = 7, Input_channels=1024, Stride=1, Kernel_size=1, N_Filtros=1024,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[27],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Global Average Pooling")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 1024, layer_outputs = np.swapaxes(np.swapaxes(activaciones[28],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=1, Output_dim_V=1, Output_dim_H = 1, Input_channels=1024, Stride=1, Kernel_size=1, N_Filtros=8,
					  layer_outputs = np.swapaxes(np.swapaxes(activaciones[29],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	return ciclos

def ZFNet_Sim(activaciones,data_buffer_1,duty_1,data_buffer_2,duty_2,word_size,frac_size):
	#print("inicia simulacion")
	ciclos = 0
	#print("Input")
	#start = timer()
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 150528, layer_outputs = np.swapaxes(np.swapaxes(activaciones[0],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=224, Output_dim_V=109, Output_dim_H = 109, Input_channels=3, Stride=2, Kernel_size=7, N_Filtros=96, layer_outputs = np.swapaxes(np.swapaxes(activaciones[1],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 279936, layer_outputs = np.swapaxes(np.swapaxes(activaciones[2],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=57, Output_dim_V=27, Output_dim_H = 27, Input_channels=96, Stride=2, Kernel_size=5, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[3],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 43264, layer_outputs = np.swapaxes(np.swapaxes(activaciones[4],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=256, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[5],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=3, N_Filtros=384, layer_outputs = np.swapaxes(np.swapaxes(activaciones[6],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_1, duty_p = duty_1, 
					  data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa Convolucional")
	ciclos += Sim_Conv(Input_dim=15, Output_dim_V=13, Output_dim_H = 13, Input_channels=384, Stride=1, Kernel_size=3, N_Filtros=256, layer_outputs = np.swapaxes(np.swapaxes(activaciones[7],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					  data_buffer_p = data_buffer_2, duty_p = duty_2, 
					  data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa MP")
	ciclos += Sim_Input_and_Pooling(Activaciones_por_ciclo = 12, N_act = 9216, layer_outputs = np.swapaxes(np.swapaxes(activaciones[8],1,3),2,3).flatten(), word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1, 
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=9216, Output_dim=4096, layer_outputs= activaciones[9], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=4096, layer_outputs= activaciones[10], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_1, duty_p = duty_1,
					data_buffer_s = data_buffer_2, duty_s = duty_2)
	#print(timer()-start)
	#start = timer()
	#print("Capa FC")
	ciclos += Sim_FC(Input_dim=4096, Output_dim=10,layer_outputs= activaciones[11], word_size = word_size, frac_size = frac_size,
					data_buffer_p = data_buffer_2, duty_p = duty_2,
					data_buffer_s = data_buffer_1, duty_s = duty_1)
	#print(timer()-start)
	#start = timer()
	return ciclos