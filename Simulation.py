from numba import cuda
from tensorflow.python.keras import backend as K
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
#####Cuda&MMU configuration##############################################################################
#Cuda
threadsperblock = 512
#Matrix Multiplication Unit dimensions
ROWS      	   = 25
COLUMNS   	   = 25
#DRAM2buffer Bandwith
acts_per_cycle = 25
#####load/save functions#################################################################################
def save_obj(obj, obj_dir ):
	with open( obj_dir + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(obj_dir):
	with open( obj_dir + '.pkl', 'rb') as f:
		return pickle.load(f)


###################################################################################################
##FUNCTION NAME: update_high_cycle_count
##DESCRIPTION:   update the count of high cycles in each cell of the buffer
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####data:   np.array representing the buffer
####stats:  np.array representing the count of high cycles in the buffer
####cycles: number of cycles since last call
###################################################################################################
@cuda.jit
def update_high_cycle_count(data,stats,cycles):
	pos = cuda.grid(1)
	if pos < stats.size and data[pos] == 1:
		stats[pos] += cycles            

###################################################################################################
##FUNCTION NAME: get_all_outputs
##DESCRIPTION:   get the activations of each layer for a given input
##OUTPUTS:       List of np.arrays with activations per layer
############ARGUMENTS##############################################################################
####model:       tf.keras.model: neural network model
####input_data:  np.array
###################################################################################################
def get_all_outputs(model, input_data, learning_phase=False):
		outputs   = [layer.output for layer in model.layers] # exclude Input
		layers_fn = K.function([model.input], outputs)
		return layers_fn([input_data])

###################################################################################################
##FUNCTION NAME: simulate_one_image
##DESCRIPTION:   simulation of one buffer for one image
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####layers:       list of layers that write in buffers.
####activations:  list of activations to write in buffers.
####config:       Dictionary with the simulation configuration.
####first_buffer: 0 if the initial write buffer is the simulated one.
###################################################################################################
def simulate_one_image(layers, activations, config, buffer, first_buffer):
	###################################################################################################
    ##FUNCTION NAME: write_in_buffer
    ##DESCRIPTION:   write the binary equivalent of activations in buffer
    ##OUTPUTS:       void
    ############ARGUMENTS##############################################################################
    ####activations: np.array of activations to write
    ####buffer:      np.array representing the buffer
    ####offset:      write address
    ###################################################################################################
	def write_in_buffer(activations, buffer, offset):
		#Function to convert integer representation to sign magnitude binary
		def integer_to_binary(integer_array):
			binary_repr_vector = np.vectorize(np.binary_repr)
			binary_array = np.where(integer_array<0,binary_repr_vector(-integer_array + 2**(config['word size']-1), width=config['word size']),
											        binary_repr_vector(integer_array, width=config['word size']))
			binary_array = np.array(list(''.join(binary_array)),dtype=int)
			return binary_array
		#Function to apply dvth mitigation common techniques
		def invert_and_shift_technique(binary_array):
			array_after_invertion = 1 - binary_array if config['invert bits'] else binary_array
			values         = np.split(array_after_invertion,len(array_after_invertion))
			shifted_values = np.roll(values,config['bitshift'])
			return np.concatenate(shifted_values)
		integer_repr = np.floor(activations*2**config['frac size']).astype(np.int32)
		binary_rep   = integer_to_binary(integer_repr)
		binary_rep_after_techniques = invert_and_shift_technique(binary_rep)
		buffer[offset:offset + len(binary_rep_after_techniques)] = binary_rep_after_techniques
	###################################################################################################
    ##FUNCTION NAME: hardware_utilization
    ##DESCRIPTION:   return procesing parameters according to the layer characteristics
    ##OUTPUTS:       Dictionary of parameters
    ############ARGUMENTS##############################################################################
    ####layer: tf.keras.layers
    ###################################################################################################
	def hardware_utilization(layer):
		parameters = {}
		if layer.__class__.__name__ in ['Conv2D','Conv1D','DepthwiseConv2D']:
			map_size                       = layer.output_shape[1] if layer.__class__.__name__ == 'Conv1D' else layer.output_shape[1]*layer.output_shape[2]
			parameters['Procesing Cycles'] = np.prod(layer.kernel_size) if layer.__class__.__name__ == 'DepthwiseConv2D' else np.prod(layer.kernel_size)*layer.input_shape[-1]
			parameters['Columns Used']     = 1 if layer.__class__.__name__ == 'DepthwiseConv2D' else min(layer.filters, COLUMNS, parameters['Procesing Cycles'])
			parameters['Rows Used']        = min(ROWS, map_size)
			parameters['Initial Delay']    = parameters['Columns Used'] + parameters['Rows Used'] + 2
			parameters['Total cycles']     = int(parameters['Initial Delay'] + parameters['Procesing Cycles']*np.ceil(map_size/parameters['Rows Used'])*np.ceil(layer.output_shape[-1]/parameters['Columns Used']))							 
		else:
			parameters['Columns Used']     = min(layer.output_shape[-1],COLUMNS)
			parameters['Rows Used']        = min(int(np.ceil(layer.output_shape[-1]/parameters['Columns Used'])),ROWS)
			parameters['Initial Delay']    = 2
			parameters['Procesing Cycles'] = layer.input_shape[-1]
			parameters['Total cycles']     = int(parameters['Initial Delay'] + parameters['Procesing Cycles']*np.ceil(layer.output_shape[-1]/(parameters['Columns Used']*parameters['Rows Used'])))
		return parameters
    ###################################################################################################
    ##FUNCTION NAME: DRAM_bypass
    ##DESCRIPTION:   called when a layer is too big for buffer so is writed in DRAM instead.
    ##OUTPUTS:       void
    ############ARGUMENTS##############################################################################
    ####buffer:       np.array representing the buffer
    ####cnn_gated:    True if CNN_gating thecnique is used
    ####layer_cycles: number of cycles required in the layer.
    ###################################################################################################
	def DRAM_bypass(buffer, cnn_gated, layer_cycles):
		if cnn_gated:
				buffer['Data'][:] = 2
		#Update Stats
		buffer['HighCyclesCount'][buffer['Data']==1] += layer_cycles
		buffer['OffCyclesCount'][buffer['Data']==2]  += layer_cycles
	###################################################################################################
    ##FUNCTION NAME: OnOff_buffer
    ##DESCRIPTION:   Turn off or on part of the buffer based in the actual layer size 
    ##OUTPUTS:       memory address for the next layer
    ############ARGUMENTS##############################################################################
    ####buffer:      np.array representing the memory
    ####out_size:    size of the actual layer
    ###################################################################################################
	def OnOff_buffer(buffer, out_size):
		#Wake up sleeping areas.
		tmp = np.random.randint(2,size=buffer['Data'].size).astype(np.int8)
		buffer['Data']   = np.where(buffer['Data']==2,tmp,buffer['Data'])
		# Get Banks partition
		buffer_divitions = list(range(0,buffer['Number of Addresses']+1,buffer['Number of Addresses']//config['Number of switchable sections']))
		# Bounds of the used area
		lower_bound = buffer['offset']
		lower_bound = max([itm for itm in buffer_divitions if itm <= buffer['offset']])
		upper_bound = min([itm for itm in buffer_divitions if itm >= (buffer['offset']+out_size)%buffer['Number of Addresses']],default=buffer['Number of Addresses'])
		if buffer['offset'] + out_size <= buffer['Number of Addresses']:
			buffer['Data'][:lower_bound*config['word size']] = 2
			buffer['Data'][upper_bound*config['word size']:] = 2
		else:
			buffer['Data'][upper_bound*config['word size']:lower_bound*config['word size']] = 2
		return upper_bound
	###################################################################################################
    ##FUNCTION NAME: write_Loop
    ##DESCRIPTION:   Loop to compute the layer procesing in the MMU
    ##OUTPUTS:       number of cycles spended	
    ############ARGUMENTS##############################################################################
    ####buffer:        np.array representing the buffer
    ####layer:         tf.keras.layer
    ####activations:   np.array of activations of the layer
    ####manual_offset: starting offset to adjust in concatenate and expand layers
    ####params:        Parameters of MMU array
    ###################################################################################################
	def write_Loop(buffer, layer, activations, manual_offset, params):
		layer_type = layer.__class__.__name__ if layer else 'CpuLayer'
		if config['CNN-Gated']:
			next_bank     = OnOff_buffer(buffer,activations.size)
		initial_state     = buffer['Data'].copy()
		wrap = True if buffer['offset'] + manual_offset + activations.size > buffer['Number of Addresses'] else False
		start_address     = (buffer['offset']+manual_offset)*config['word size']
		end_address       = start_address + activations.size*config['word size']
		#Working with a copy of the data
		high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint32)	
		data              = np.take(buffer['Data'],range(start_address,end_address),mode='wrap')
		#Cuda parameter
		blockspergrid = (high_cycles_count.size + (threadsperblock - 1)) // threadsperblock
		#Update stats after the initial delay
		cycles = params['Initial Delay']
		update_high_cycle_count[blockspergrid, threadsperblock](data,high_cycles_count,params['Initial Delay'])
		#Acceleration of GPU Memory Transfer using small representation
		tmp_counter = 0
		if params['Procesing Cycles'] < 255:
			tmp_high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint8)
			max_value = 255
		else:
			tmp_high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint16)
			max_value = 65535
		if layer_type in ['Conv2D','Conv1D','DepthwiseConv2D']:
			for filter_patch in range(0,layer.output_shape[-1],params['Columns Used']):
				for activations_patch in range(0,activations.shape[0],params['Rows Used']):
					cycles      += params['Procesing Cycles']
					tmp_counter += params['Procesing Cycles']   
					update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
					# For the simulation speed purposes its assumed that the accelerator wait for a entire PE array activations (125) to be deposited in the local FIFO buffers and write all them in the same cycle
					processed_outputs = activations[activations_patch:activations_patch+params['Rows Used'],filter_patch:filter_patch+params['Columns Used']]
					act_offset = activations_patch*layer.output_shape[-1]
					for out_slice in processed_outputs:
						write_in_buffer(out_slice,data,(act_offset+filter_patch)*config['word size'])
						act_offset += layer.output_shape[-1]
					# Preventing possible overflow
					if tmp_counter + params['Procesing Cycles'] > max_value:
						high_cycles_count += tmp_high_cycles_count
						tmp_counter = 0
						tmp_high_cycles_count[:] = 0
		elif layer_type == 'Dense':
			for output_patch in range(0,activations.shape[0],params['Columns Used']*params['Rows Used']):
				cycles      += params['Procesing Cycles']
				tmp_counter += params['Procesing Cycles']
				update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
				# Its assumed that the accelerator can compute a neuron in each PE in parallel, again for simulation speed purposes the (125) outputs of a patch are writed in the same cycle they are processed.
				processed_outputs = activations[output_patch:output_patch+params['Columns Used']*params['Rows Used']]
				write_in_buffer(processed_outputs,data,output_patch*config['word size'])
				# Preventing possible overflow
				if tmp_counter + params['Procesing Cycles'] > max_value:
					high_cycles_count += tmp_high_cycles_count
					tmp_counter = 0
					tmp_high_cycles_count[:] = 0
		else:
			for activations_patch in range(0, activations.size, acts_per_cycle):
				cycles += params['Procesing Cycles']
				tmp_counter += params['Procesing Cycles']
				update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
				processed_outputs = activations[activations_patch:activations_patch + acts_per_cycle]
				write_in_buffer(processed_outputs,data,activations_patch*config['word size'])
				# Preventing possible overflow
				if tmp_counter + params['Procesing Cycles'] > max_value:
					high_cycles_count += tmp_high_cycles_count
					tmp_counter = 0
					tmp_high_cycles_count[:] = 0
		#Pass Leftovers
		high_cycles_count += tmp_high_cycles_count
		#Update buffer stats
		if wrap:
			buffer['HighCyclesCount'][start_address:] += high_cycles_count[:buffer['Number of Addresses']*config['word size']-start_address]
			buffer['HighCyclesCount'][:end_address-buffer['Number of Addresses']*config['word size']] += high_cycles_count[buffer['Number of Addresses']*config['word size']-start_address:]
			buffer['HighCyclesCount'][end_address-buffer['Number of Addresses']*config['word size']:start_address][buffer['Data'][end_address-buffer['Number of Addresses']*config['word size']:start_address]==1] += cycles
			buffer['Data'][start_address:] = data[:buffer['Number of Addresses']*config['word size']-start_address]
			buffer['Data'][:end_address-buffer['Number of Addresses']*config['word size']] = data[buffer['Number of Addresses']*config['word size']-start_address:]
			buffer['OffCyclesCount'][buffer['Data']==2] += cycles
		else:
			buffer['HighCyclesCount'][start_address:end_address] += high_cycles_count
			buffer['HighCyclesCount'][:start_address][buffer['Data'][:start_address]==1] += cycles 
			buffer['HighCyclesCount'][end_address:][buffer['Data'][end_address:]==1]     += cycles
			buffer['Data'][start_address:end_address] = data
			buffer['OffCyclesCount'][buffer['Data']==2] += cycles
		buffer['Flips'][buffer['Data'] != initial_state] += 1
		if config['CNN-Gated']:
			buffer['offset'] = next_bank
		return cycles
	##############Layer Simulation#######################################################################
	#Clasic Convolution
	def sim_conv(buffer, layer, layer_outputs, write_buffer_id):
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles']
			return params['Total cycles']
		#Convert acts from (B,H,W,C) to (H*W,C)
		activations = layer_outputs[0].reshape(-1,layer_outputs[0].shape[-1])
		layer_size = activations.size*config['word size']
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			DRAM_bypass(buffer,config['CNN-Gated'],params['Total cycles'])
			return params['Total cycles']
		else:
			cycles = write_Loop(buffer,layer,activations,0,params)
			return cycles
	# Dense layer
	def sim_FC(buffer, layer, layer_outputs, write_buffer_id):
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles']
			return params['Total cycles']
		# Convert acts from (B,N) to (N) 
		activations = layer_outputs[0]
		layer_size = activations.size*config['word size']
		cycles = write_Loop(buffer,layer,activations,0,params)
		return cycles
	#layer computed in cpu (Input,embedding)
	def sim_CPULayer(buffer, layer, layer_outputs, write_buffer_id):
		params = {'Initial Delay': 0, 'Procesing Cycles': 1}
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			buffer['HighCyclesCount'][buffer['Data']==1] += int(np.ceil(layer_outputs.size/acts_per_cycle))
			buffer['OffCyclesCount'][buffer['Data']==2]  += int(np.ceil(layer_outputs.size/acts_per_cycle))
			return int(np.ceil(layer_outputs.size/acts_per_cycle))
		#Convert acts from (B,H,W,C) to (H*W,C)
		activations = layer_outputs[0].flatten()
		layer_size = activations.size*config['word size']
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			DRAM_bypass(buffer,config['CNN-Gated'],int(np.ceil(layer_outputs.size/acts_per_cycle)))
			return int(np.ceil(layer_outputs.size/acts_per_cycle))
		else:
			cycles = write_Loop(buffer,layer,activations,0,params)
			return cycles
	#Expand (SqueezeNet layer)
	def sim_expand(buffer, layers, layer_outputs, write_buffer_id):
		params1x1 = hardware_utilization(layers[0])
		params3x3 = hardware_utilization(layers[1])
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			buffer['HighCyclesCount'][buffer['Data']==1] += params1x1['Total cycles'] + params3x3['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params1x1['Total cycles'] + params3x3['Total cycles']
			return params1x1['Total cycles'] + params3x3['Total cycles']
		# Convert acts from (B,H,W,C) to (H*W,C) 
		out_channels = layer_outputs[0].shape[-1]
		activations  = layer_outputs[0].reshape(-1,out_channels)
		activations1x1 = activations[:,:out_channels//2]
		activations3x3 = activations[:,out_channels//2:]
		layer_size = (activations1x1.size + activations3x3.size)*config['word size']
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			DRAM_bypass(buffer,config['CNN-Gated'],params1x1['Total cycles'] + params3x3['Total cycles'])
			return params1x1['Total cycles'] + params3x3['Total cycles']
		else:
			cycles =  write_Loop(buffer,layers[0],activations1x1,0,params1x1)
			cycles += write_Loop(buffer,layers[1],activations3x3,activations1x1.size if (not config['CNN-Gated']) and (config['write mode'] == 'default') else 0,params3x3)
			return cycles
	#Concatenation of Conv (DenseNet layer)
	def sim_concat_conv(buffer, layer, layer_outputs, write_buffer_id):
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
			return params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
		#Convert acts from (B,H,W,C) to (H*W,C) and (H*W*C)
		activations1 = layer_outputs[0][0].reshape(-1,layer_outputs[0][0].shape[-1])
		activations2 = layer_outputs[1][0].flatten()
		layer_size1 = activations1.size*config['word size']
		layer_size2 = activations2.size*config['word size']
		#Check if layer must be bypassed to DRAM
		if layer_size1 + layer_size2 > buffer['Number of Addresses']*config['word size']:
			DRAM_bypass(buffer,config['CNN-Gated'],params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle)) )
			return params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
		else:
			cycles = write_Loop(buffer,layer,activations1,0,params)
			cycles += write_Loop(buffer,None,activations2,activations1.size if (not config['CNN-Gated']) and (config['write mode'] == 'default') else 0,{'Initial Delay': 0,'Procesing Cycles': 1})
			return cycles
	#########Dispatcher to the simulation of the corresponding layer#####################################
	def handle_layer(buffer, layer, layer_outputs, out_buffer_id):
		if isinstance(layer_outputs,list):
			return sim_concat_conv(buffer, layer, layer_outputs, out_buffer_id)
		elif isinstance(layer, np.ndarray):
			return sim_expand(buffer, layer, layer_outputs, out_buffer_id)
		elif layer.__class__.__name__ in ['Conv2D','Conv1D','DepthwiseConv2D']:
			return sim_conv(buffer, layer, layer_outputs, out_buffer_id)
		elif layer.__class__.__name__ == 'Dense':
			return sim_FC(buffer, layer, layer_outputs, out_buffer_id)
		else:
			return sim_CPULayer(buffer, layer, layer_outputs,out_buffer_id)
	#########Simulation layer by layer####################################################################
	cycles = 0
	# We only simulate buffer 2 (buffer 1 behaves statistically equal)
	if first_buffer == 1 and config['CNN-Gated']:
		buffer['Data'][:] = 2
	for index,layer in enumerate(layers):
		tmp = cycles
		cycles += handle_layer(buffer,layer,activations[index],(first_buffer+index)%2)
	return cycles
###################################################################################################
##FUNCTION NAME: buffer_simulation
##DESCRIPTION:   Dimulation of one buffer and collect stats associated.
##OUTPUTS:       Dictionary of stats and total number of cycles
############ARGUMENTS##############################################################################
####network:            tf.keras.model: model of neural network 
####dataset:            Dataset iterator to be used as inference
####integer_bits:       Number of bit for integer part of activations
####fractional_bits:    Number of bit for fractional part of activations
####bit_invertion:      True if bit invertion is desired (with granularity of 2 images)
####bit_shifting:       True if bit shifting is desired (with granularity of 2 images)
####write_mode:         True if a rotation of writing address is desired (by default with CNN_gating)
####CNN_gating:         True if CNN_gating is applied
####buffer_size:        Size of the buffer (in bytes)
####start_from:         Index of image in dataset to start from (in case of resumed execution)
####results_dir:        Directory for saving the results, False if not saving required
####layer_indexes:      List of index of layer processed
####activation_indixes: List of index of layer with the results
####network_type:       'None' or 'MobileNet'
###################################################################################################
def buffer_simulation(network, dataset, samples, integer_bits, fractional_bits, bit_invertion,
                      bit_shifting, write_mode, CNN_gating, buffer_size, start_from,
					  layer_indexes, activation_indixes, results_dir = False, network_type = None):
	###############Stats vars#############################################################################################################
	# Get initial state
	if start_from == 0:
		#Context
		config = {}
		# Data Representation
		config['word size']   = (1+integer_bits+fractional_bits)
		config['frac size']   = fractional_bits
		# Duty Techniques
		config['invert bits'] = 0
		config['bitshift']    = 0
		config['CNN-Gated']   = CNN_gating
		config['write mode']  = write_mode
		config['Number of switchable sections'] = 8
		#Buffer
		buffer = {}
		buffer['Number of Addresses'] = buffer_size//2
		buffer['Data']                = np.zeros(buffer_size*8,dtype=np.int8)
		buffer['HighCyclesCount']     = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['OffCyclesCount']      = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['LowCyclesCount']      = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['Flips']               = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['offset']              = 0
		# Initial variables
		cycles                        = 0
	else:
		buffer  = load_obj(results_dir + 'buffer')
		cycles  = load_obj(results_dir + 'cycles')
		config  = load_obj(results_dir + 'config')
	print('buffer sections: ',list(range(0,buffer['Number of Addresses']+1,buffer['Number of Addresses']//config['Number of switchable sections'])))
	################Simulation Loop########################################################################################################
	layers = [np.take(network.layers,index) for index in layer_indexes]
	X = [x for x,y in dataset]
	print('Simulation Started, time:',datetime.now().strftime("%H:%M:%S"),'cycles: ',cycles, 'offset: ',buffer['offset'])
	for index in range(start_from,samples):
		# Simulate the next image
		activacions = get_all_outputs(network,X[index])
		activacions = [activacions[itm] if type(itm) != tuple else [activacions[subitm] for subitm in itm] for itm in activation_indixes]
		if network_type == 'MobileNet':
			activacions[-1] = activacions[-1].reshape((1,1,1,activacions[-1].size)) 
		cycles += simulate_one_image(layers,activacions,config, buffer, index%2)
		buffer['LowCyclesCount'] = cycles -  buffer['HighCyclesCount']  -  buffer['OffCyclesCount']
		# Update duty strategies
		if index % 2 == 0 and bit_invertion:
			config['invert bits'] = 1 - config['invert bits']
		if index % 2 == 0 and bit_shifting:
			config['bitshift'] = (config['bitshift'] + 1) % config['word size']
		# Save Results
		if index % 25 == 0 and results_dir:
			save_obj(buffer,results_dir+'buffer %s images'%index)
			save_obj(cycles, results_dir+'cycles %s images'%index)
			save_obj(config, results_dir+'config %s images'%index)
		if results_dir:
			save_obj(buffer, results_dir+'buffer')
			save_obj(cycles, results_dir+'cycles')
			save_obj(config, results_dir+'config')
		print('procesed images:',index,' time:',datetime.now().strftime("%H:%M:%S"),'cycles: ',cycles, 'offset: ',buffer['offset'])
	return buffer,cycles