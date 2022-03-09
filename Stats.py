import numpy as np
import pandas as pd
import tensorflow as tf
import gc
from tensorflow.python.keras import backend as K
from Nets import get_neural_network_model


def GenerateFaultsList(shape,locs,error_mask):
	# Decodes the mask of faults to the specific error due to 0 and 1 static value
	def DecodeMask(mask):
		static1Error  = int("".join(mask.replace('x','0')),2)
		static0Error  = int("".join(mask.replace('x','1')),2)
		return [static0Error,static1Error]
	positionList   = []
	faultList      = []
	if len(shape) == 1:
		for address,mask in zip(locs,error_mask):
			if address < shape[0] - 1:
				positionList.append([address])
				faultList.append(DecodeMask(mask))
	elif len(shape) == 2:
		for address,mask in zip(locs,error_mask):
			if address < shape[0]*shape[1] - 1:
				Ch1  = address//shape[0]
				Ch2  = (address - Ch1*shape[0])//shape[1]
				positionList.append([Ch1,Ch2])
				faultList.append(DecodeMask(mask))
	else:
		for address,mask in zip(locs,error_mask):
			if address < shape[0]*shape[1]*shape[2] - 1:
				filt   = address//(shape[0]*shape[1]*shape[2])
				row    = (address - filt*shape[0]*shape[1]*shape[2])//(shape[1]*shape[2])
				col    = (address - filt*shape[0]*shape[1]*shape[2] - row*shape[1]*shape[2])//shape[2]
				Amap   = address  - filt*shape[0]*shape[1]*shape[2] - row*shape[1]*shape[2] - col*shape[2]
				positionList.append([row,col,Amap,filt])
				faultList.append(DecodeMask(mask))
	NumberOfFaults = len(positionList)
	return tf.convert_to_tensor(positionList),tf.convert_to_tensor(faultList), NumberOfFaults

def introduce_faults_in_weights(tensor,positionList,faultList,intSize,fractSize):
	def apply_fault(tensor,faults):
		Ogdtype = tensor.dtype
		shift   = 2**(intSize+fractSize-1)
		factor  = 2**fractSize
		tensor  = tf.cast(tensor*factor,dtype=tf.int32)
		tensor  = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
		tensor  = tf.bitwise.bitwise_and(tensor,faults[:,0])
		tensor  = tf.bitwise.bitwise_or(tensor,faults[:,1])
		tensor  = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
		tensor  = tf.cast(tensor/factor,dtype = Ogdtype)
		return tensor
	affected_values = tf.gather_nd(tensor,positionList)
	new_values = apply_fault(affected_values,faultList)
	tensor = tf.tensor_scatter_nd_update(tensor, positionList, new_values)
	return tensor

def weight_quantization(model, frac_bits, int_bits):
	def Quantization(tensor):
		factor = 2.0**frac_bits
		max_Value = ((1 << (frac_bits+int_bits)) - 1)/factor
		min_Value = -max_Value - 1
		tensor = tf.round(tensor*factor) / factor
		tensor = tf.math.minimum(tensor,max_Value)   # Upper Saturation
		tensor = tf.math.maximum(tensor,min_Value)   # Lower Saturation
		return tensor 
	for layer in model.layers:
		weights = layer.get_weights()
		if weights:
			qWeights    = [Quantization(itm) for itm in weights]
			layer.set_weights(qWeights)

###################################################################################################
##FUNCTION NAME: check_accuracy_and_loss
##DESCRIPTION:   Evaluation of Accuracy and/or Loss under certain assumptions
##OUTPUTS:       accuracy and/or loss
############ARGUMENTS##############################################################################
####architecture:     one of the followings: 'AlexNet','VGG16','PilotNet',
####                  'MobileNet','ZFNet','SqueezeNet','SentimentalNet','DenseNet'
####test_dataset:     dataset iterable
####wgt_dir:          Directory of weights
####act_frac_size:    Fractional part activation size
####act_int_size:     Integer part activation size
####wgt_frac_size:    Fractional part weights size
####wgt_int_size:     Integer part weights size
####input_shape:      input dimensions (3 channels assumed)
####output_shape:     output dimensions
####faulty_addresses: List of addresses with faults in memory
####masked_faults:    List of faults types for each address in faulty_addresses
####aging_active:     True if aging effect should be considerated
####batch_size:       Size of batch in inference
####weights_faults:   True if faults in weight buffer are applied instead of activation buffer
###################################################################################################
def check_accuracy_and_loss(architecture, test_dataset, wgt_dir, act_frac_size, act_int_size, wgt_frac_size, wgt_int_size, input_shape, output_shape,
						 faulty_addresses = [], masked_faults = [], aging_active = False, batch_size = 1, verbose = 1, weights_faults = False):
	qNet = get_neural_network_model(architecture,input_shape,output_shape,faulty_addresses,masked_faults,aging_active=aging_active,
								 word_size=(1+act_frac_size+act_int_size), frac_size=act_frac_size, batch_size = batch_size)
	#Load Weights
	qNet.load_weights(wgt_dir).expect_partial()
	#Quantize Weights
	weight_quantization(model = qNet, frac_bits = wgt_frac_size, int_bits = wgt_int_size)
	if weights_faults:
		weights = qNet.get_weights()
		for index,itm in enumerate(weights):
			positionList,faultList,NumberOfFaults = GenerateFaultsList(shape=itm.shape,locs=faulty_addresses,error_mask=masked_faults)
			if NumberOfFaults > 0:
				weights[index] = introduce_faults_in_weights(itm,positionList,faultList,wgt_int_size,wgt_frac_size)
		qNet.set_weights(weights)
	# Params
	if architecture == 'Sentimental':
		loss = 'binary_crossentropy'
	else:
		loss = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
	if architecture == 'PilotNet':
		qNet.compile(optimizer=optimizer, loss='mse',)
		loss = qNet.evaluate(test_dataset,verbose=verbose)
		outputs = (None,loss)
	else:
		qNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
		(loss,acc) = qNet.evaluate(test_dataset,verbose=verbose)
		outputs = (loss,acc)
	# Cleaning Memory
	del qNet
	gc.collect()
	K.clear_session()
	tf.compat.v1.reset_default_graph()
	return outputs


###################################################################################################
##FUNCTION NAME: activation_stats
##DESCRIPTION:   Print stats of activations
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####network:           tf.keras.model
####test_dataset:      dataset iterable
####frac_bits:         number of bits fractional activation part
####int_bits:          number of bits integer activation part
####dataset_size:      number of batches in dataset
###################################################################################################
def activation_stats(network, test_dataset, frac_bits, int_bits, dataset_size):
	def get_all_outputs(model, input_data, learning_phase=False):
		outputs = [layer.output for layer in model.layers] # exclude Input
		layers_fn = K.function([model.input], outputs)
		return layers_fn([input_data])

	max_value   = ((1 << (frac_bits+int_bits)) - 1)/(2.0**frac_bits)
	min_value   = -max_value - 1
	iterator   = iter(test_dataset)
	minimum_buffer  = 999
	maximum_buffer  = -999
	buffer_means   = []
	buffer_indexes = []
	saturated_buffer_count = buffer_activation_count = 0
	for index in range(len(network.layers)):
		if index == len(network.layers) - 1:
			buffer_indexes.append(index)
		elif network.layers[index+1].__class__.__name__ in ['Conv2D','Conv1D','MaxPooling2D','MaxPooling1D','GlobalAveragePooling2D','Dense','AveragePooling2D','DepthwiseConv2D']:
			buffer_indexes.append(index)
		elif network.layers[index].__class__.__name__ in ['Conv2D','Conv1D','Dense','DepthwiseConv2D']:
			MMU_indexes.append(index)
	index  = 1
	while index <= dataset_size:
		image       = next(iterator)[0]
		activations = get_all_outputs(network,image)
		buffer_acts = [activations[i] for i in buffer_indexes]
		buffer_acts = np.concatenate( [itm.flatten() for itm in buffer_acts] , axis=0 )
		tmp3 = np.max(buffer_acts)
		tmp4 = np.min(buffer_acts)
		if tmp3 > maximum_buffer:
			maximum_buffer = tmp3
		if tmp4 < minimum_buffer:
			minimum_buffer = tmp4
		buffer_means.append(np.mean(buffer_acts))
		saturated_buffer_count  += np.sum(buffer_acts > max_value) + np.sum(buffer_acts < min_value)
		buffer_activation_count += len(buffer_acts)
		index = index + 1
	print('mean value (Buffer):',np.mean(buffer_means))
	print('maximum (Buffer):',maximum_buffer)
	print('minimum (Buffer):',minimum_buffer)
	print('saturation ratio (Buffer):',saturated_buffer_count/buffer_activation_count)

###################################################################################################
##FUNCTION NAME: quantization_effect
##DESCRIPTION:   Test the effect of quantization in accuracy for different distribution of bits
##OUTPUTS:       pandas dataframe 
############ARGUMENTS##############################################################################
#### same arguments than check_accuracy_and_loss
#### activations-bits:   total bytes per activation
###################################################################################################
def quantization_effect(architecture, dataset, wgt_dir, input_shape, output_shape, batch_size, activations_bits = 16, verbose=0):
	df = pd.DataFrame({'Experiment':[],'bits':[],'acc':[],'loss':[]})
	bits = range(0,activations_bits-1) # bits to test in quantized parts.
	print('Varying the number of fractional part bits in the activations:')
	for bit in bits:
		loss, acc = check_accuracy_and_loss(architecture, dataset, wgt_dir, act_frac_size = bit, act_int_size = activations_bits, wgt_frac_size = activations_bits, wgt_int_size = activations_bits, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print('number of bits:',bit,' results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Varying the number of fractional part bits in the weights:')
	for bit in bits:
		loss, acc = check_accuracy_and_loss(architecture, dataset, wgt_dir, act_frac_size = activations_bits, act_int_size = activations_bits, wgt_frac_size = bit, wgt_int_size = activations_bits, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print('number of bits:',bit,' results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Weights fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Varying the number of integer part bits in the activations:')
	for bit in bits:
		loss, acc = check_accuracy_and_loss(architecture, dataset, wgt_dir, act_frac_size = activations_bits, act_int_size = bit, wgt_frac_size = activations_bits, wgt_int_size = activations_bits, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print('number of bits:',bit,' results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation integer part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Varying the number of integer part bits in the weights:')
	for bit in bits:
		loss, acc = check_accuracy_and_loss(architecture, dataset, wgt_dir, act_frac_size = activations_bits, act_int_size = activations_bits, wgt_frac_size = activations_bits, wgt_int_size = bit, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print('number of bits:',bit,' results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Weights integer part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	return df


###################################################################################################
##FUNCTION NAME: get_read_and_writes
##DESCRIPTION:   get number of reads and writes by each cell in buffer memory
##OUTPUTS:       Dictionary of writes and reads
############ARGUMENTS##############################################################################
####network:          tf.keras.model
####layer_indices:    list of index of layers of interest
####addresses:        number of address in the buffer
####samples:          number of images to simulate
####CNN_gating:       True if CNN_gating technique is considered
####network_name:     'None' o 'MobileNet'
###################################################################################################
def get_read_and_writes(network,layer_indices,addresses,samples,CNN_gating=False, network_name = False):
	PE_columns = 25
	PE_rows    = 25
	Data = {}
	Data['Writes'] = np.zeros(addresses,dtype=np.int64)
	Data['Reads']  = np.zeros(addresses,dtype=np.int64)
	Data['offset'] = 0
	def get_next_address(outsize):
		buffer_divitions = list(range(0,addresses+1,addresses//8))
		next_address = 0 if not CNN_gating else min([itm for itm in buffer_divitions if itm >= (Data['offset']+outsize)%addresses],default=addresses) 
		Data['offset'] = next_address
	def Buffer_Writing(layer):
		out_size = np.prod(layer.output_shape[0][1:]).astype(np.int64) if type(layer.output_shape[0])==tuple else np.prod(layer.output_shape[1:]).astype(np.int64)
		if out_size > addresses:
			return
		actual_values = np.take(Data['Writes'],range(Data['offset'],Data['offset']+out_size),mode='wrap')
		np.put(Data['Writes'],range(Data['offset'],Data['offset']+out_size),1+actual_values,mode='wrap')
		get_next_address(out_size)
	def Buffer_Reading(layer):
		if layer.__class__.__name__ in ['MaxPooling2D','GlobalAveragePooling2D','MaxPooling1D','AveragePooling2D','BatchNormalization']:
			in_size = np.prod(layer.input_shape[1:]).astype(np.int64)
			if in_size > addresses:
				return
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+in_size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+in_size),1+actual_values,mode='wrap')
		elif len(layer.input_shape) == 4:
			#Convolution
			(input_height,input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			if padding == 'valid':
				input_map = np.zeros((input_height,input_width,input_channels),dtype=np.int64)
			else:
				height_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				width_zero_pads = np.ceil(layer.kernel_size[1]/2).astype(np.int64)
				input_map = np.zeros((input_height+height_zero_pads,input_width+width_zero_pads,input_channels),dtype=np.int64)
			(output_height,output_width,output_channels) = layer.output_shape[1:]
			#Iterate over outputs and update the reads from Activation Map
			Kernel_X1 = 0
			Kernel_X2 = layer.kernel_size[0]
			Kernel_Y1 = 0
			Kernel_Y2 = layer.kernel_size[1]
			(Stride_X,Stride_Y)  = layer.strides
			for y in range(output_height):
				for x in range(output_width):
					input_map[Kernel_Y1:Kernel_Y2,Kernel_X1:Kernel_X2,:] += 1
					Kernel_X1 += Stride_X
					Kernel_X2 += Stride_X
				Kernel_X1 = 0
				Kernel_X2 = layer.kernel_size[0]
				Kernel_Y1 += Stride_Y
				Kernel_Y2 += Stride_Y
			if layer.__class__.__name__ in ['Conv2D']: 
				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(height_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -height_zero_pads//2,-1)),0)
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),1)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),1)
			input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
		elif len(layer.input_shape) == 3:
			#Convolution 1D
			(input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			if padding == 'valid':
				input_map = np.zeros((input_width,input_channels),dtype=np.int64)
			else:
				width_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				input_map = np.zeros((input_width+width_zero_pads,input_channels),dtype=np.int64)
			(output_width,output_channels) = layer.output_shape[1:]
			#Iterate over outputs and update the reads from Activation Map
			Kernel_X1 = 0
			Kernel_X2 = layer.kernel_size[0]
			Stride_X  = layer.strides[0]
			for x in range(output_width):
				input_map[Kernel_X1:Kernel_X2,:] += 1
				Kernel_X1 += Stride_X
				Kernel_X2 += Stride_X
			if layer.__class__.__name__ in ['Conv1D']: 
				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),0)
			input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
		else:
			#FC
			input_size  = layer.input_shape[-1]
			output_size = layer.output_shape[-1]
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_size),input_size*np.ceil(output_size/(PE_columns*PE_rows)).astype(np.int64)+actual_values,mode='wrap')
	def sim_layer(layer,bid):
		if bid == 0:
			Buffer_Writing(layer)
		else:
			Buffer_Reading(layer)
	def sim_concat(layer,bid):
		if bid == 0:
			Buffer_Writing(layer[0])
			Buffer_Writing(layer[1])
		else:
			Buffer_Reading(layer[0])
	def sim_expand(layer,bid):
		if bid == 0:
			Buffer_Writing(layer[0])
			Buffer_Writing(layer[1])
		else:
			Buffer_Reading(layer[0])
			Buffer_Reading(layer[1])
	def handle_layer(layer,bid):
		if isinstance(layer,np.ndarray) and network_name == 'DenseNet':
			return sim_concat(layer, bid)
		elif isinstance(layer, np.ndarray) and network_name == 'SqueezeNet':
			return sim_expand(layer, bid)
		else:
			return sim_layer(layer,bid)
	def simulate_single_inference(layers,first_buffer):
		if first_buffer == 0:
			#print('procesando: ',layers[0].name)
			Buffer_Writing(layers[0])
			#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
			for index,layer in enumerate(layers[1:]):
				#print('procesando: ',layer.name)
				handle_layer(layer,(first_buffer+index+1)%2)
				#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
		else:
			for index,layer in enumerate(layers[1:]):
				#print('procesando: ',layer.name)
				handle_layer(layer,(first_buffer+index)%2)
				#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
	layers = [np.take(network.layers,index) for index in layer_indices]
	for image in range(samples):
		if (image+1)%25 == 0:
			print('procesados: ',image+1)
		simulate_single_inference(layers,image%2)
	return Data