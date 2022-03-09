from tensorflow.keras.layers import (Activation, AveragePooling2D, BatchNormalization, Cropping2D,
                                     Concatenate, Conv1D, Conv2D, Dense, DepthwiseConv2D, Dropout,
                                     Embedding, Flatten, GlobalAveragePooling2D, Lambda, MaxPool2D,
                                     MaxPooling1D, ReLU, Reshape, ZeroPadding2D)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np

###################################################################################################
##FUNCTION NAME: get_neural_network_model
##DESCRIPTION:   Build the required Neural Network 
##OUTPUTS:       tf.keras.Model
############ARGUMENTS##############################################################################
####architecture:     Name of the network; one of the following: 'AlexNet','VGG16','PilotNet',
####                  'MobileNet','ZFNet','SqueezeNet','SentimentalNet','DenseNet'.
####input_shape:      inputs height and width (3 channels assumed).
####output_shape:     Number of classes.
####faulty_addresses: List of addresses with faults.
####masked_faults:    List of faults for the given addresses in faulty_addresses.
####quantization:     True for a quantized version of the network.
####aging_active:     True if including the aging effect in memory is desired.
####word_size:        Number of bits for activations.
####frac_size:        Number of bits for fractional part of each activation.
####batch_size:       inference batch size.
###################################################################################################
def get_neural_network_model(architecture, input_shape, output_shape, faulty_addresses = [],
                         	 masked_faults = [], quantization = True, aging_active = True,
						     word_size=None, frac_size=None, batch_size = 1):	 
    ###############################################################################################
    ##FUNCTION NAME: Quantization
    ##DESCRIPTION:   Layer to quantize the values to the magnitude sign format when needed
    ##OUTPUTS:       Quantized tensor
    ############ARGUMENTS##########################################################################
    ####tensor:    Tensorflow tensor
    ####active:    False for simple bypass of the signal
    ###############################################################################################
	def Quantization(tensor, active = True):
		if active:
			factor = 2.0**frac_size
			max_value = ((1 << (word_size-1)) - 1)/factor
			min_value = -max_value
			tensor = tf.round(tensor*factor) / factor
			tensor = tf.math.minimum(tensor,max_value)   # Upper Saturation
			tensor = tf.math.maximum(tensor,min_value)   # Lower Saturation
		return tensor 
	###############################################################################################
    ##FUNCTION NAME: Aging
    ##DESCRIPTION:   Layer to apply the aging effect to the previous layer according to the buffer
    ##OUTPUTS:       Tensor with affected values due to aging
    ############ARGUMENTS##########################################################################
    ####tensor:     Tensorflow tensor
    ####index_list: tensor of index of affected values
    ####mod_list:   tensor of operators used to affect the list of index in a desired way
    ####active:     False for simple bypass of the signal
    ###############################################################################################
	def Aging(tensor, index_list, mod_list, active):
		def ApplyFault(tensor,faults):
			Ogdtype = tensor.dtype
			shift   = 2**(word_size-1)
			factor  = 2**frac_size
			tensor  = tf.cast(tensor*factor,dtype=tf.int32)
			tensor  = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
			tensor  = tf.bitwise.bitwise_and(tensor,faults[:,0])
			tensor  = tf.bitwise.bitwise_or(tensor,faults[:,1])
			tensor  = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
			tensor  = tf.cast(tensor/factor,dtype = Ogdtype)
			return tensor		
		if active:
			affected_values = tf.gather_nd(tensor,index_list)
			newValues = ApplyFault(affected_values,mod_list)
			tensor = tf.tensor_scatter_nd_update(tensor, index_list, newValues)
		return tensor
	###############################################################################################
    ##FUNCTION NAME: generate_index_list
    ##DESCRIPTION:   generate a list of index and their corresponding modify operartors, 
    ##               acording to the equivalence beetween memory addresses and layer activations
    ##OUTPUTS:       tensor of index, tensor of operands, number of activations to modify
    ############ARGUMENTS##########################################################################
    ####shape:       layer shape
    ############################################################################################### 
	def generate_index_list(shape):
		# Decodes the mask of faults to two operators to apply aging of 0 or 1 logic value
		def DecodeMask(mask):
			op_0  = int("".join(mask.replace('x','1')),2)
			op_1  = int("".join(mask.replace('x','0')),2)
			return [op_0,op_1]
		index_list    = []
		mod_list      = []
		if len(shape) == 1:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]-1:
						index_list.append([index,address])
						mod_list.append(DecodeMask(mask))
		elif len(shape) == 2:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1] - 1:
						Ch1  = address//shape[0]
						Ch2  = (address - Ch1*shape[0])//shape[1]
						index_list.append([index,Ch1,Ch2])
						mod_list.append(DecodeMask(mask))
		else:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1]*shape[2] - 1:
						act_map = address//(shape[0]*shape[1])
						row     = (address - act_map*shape[0]*shape[1])//shape[1]
						col     = address - act_map*shape[0]*shape[1] - row*shape[1]
						index_list.append([index,row,col,act_map])
						mod_list.append(DecodeMask(mask))
		faults_count = len(index_list)
		return tf.convert_to_tensor(index_list),tf.convert_to_tensor(mod_list), faults_count
	###############################################################################################
    ##FUNCTION NAME: add_custom_layers
    ##DESCRIPTION:   Wrapper of Aging and Quantization layers
    ##OUTPUTS:       sequential model with the corresponding layers added
    ############ARGUMENTS##########################################################################
    ####input_layer:          sequential layer
    ####aging_layer:          True if a aging layer is needed
    ####quantization_layer:   True if a quantization layer is needed
    ####aging_active:         True if active aging effect is desired
    ###############################################################################################
	def add_custom_layers(input_layer, aging_layer, quantization_layer=True, aging_active = False):
		x = input_layer
		if quantization_layer:
			quantization_arguments = {'active':quantization}
			x = Lambda(Quantization, arguments = quantization_arguments)(input_layer)
		if aging_layer:
			dims = x.shape.ndims if x.__class__.__name__ == 'KerasTensor' else x.output_shape.ndim
			index_list,mod_list,faults_count = generate_index_list(shape=x.shape[1:])
			aging_arguments = {'index_list' : tf.identity(index_list),
			                   'mod_list' : tf.identity(mod_list),
				               'active': tf.identity(faults_count and aging_active)}
			x = Lambda(Aging, arguments = aging_arguments)(x)
		return x
	if   aging_active == True:  aging_active = [True]*1000
	elif aging_active == False: aging_active = [False]*1000
	#AlexNet
	if architecture=='AlexNet':
		input_layer = tf.keras.Input(input_shape)
		x = add_custom_layers(input_layer,aging_layer=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),name='Conv1')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2],quantization_layer=False)
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same",name='Conv2')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[4],quantization_layer=False)
		x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same",name='Conv3')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[5])
		x = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv4')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[6])
		x = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv5')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[8],quantization_layer=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[9],quantization_layer=False)	
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[10],quantization_layer=False)		
		x = Dense(output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = tf.keras.activations.softmax(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture=='VGG16':
		input_layer = tf.keras.Input(input_shape)
		x = add_custom_layers(input_layer,aging_layer=True,aging_active = aging_active[0])
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1],quantization_layer=False)
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2],quantization_layer=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[3],quantization_layer=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[4],quantization_layer=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[5],quantization_layer=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[6],quantization_layer=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[7],quantization_layer=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[8],quantization_layer=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[9],quantization_layer=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[10],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[11],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[12],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[13],quantization_layer=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[14],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[15],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[16],quantization_layer=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[17],quantization_layer=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[18],quantization_layer=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[19],quantization_layer=False)
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[20],quantization_layer=False)
		x = Dense(output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = tf.keras.activations.softmax(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'PilotNet':
		def preprocess(image):  # preprocess image
			return tf.image.resize(image, (200, 66))
		input_layer = tf.keras.Input(input_shape)
		x = Cropping2D(cropping=((50,20), (0,0)))(input_layer)
		x = Lambda(preprocess)(x)
		x = Lambda(lambda x: (x/ 127.0 - 1.0))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[0])
		x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2))(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2])
		x = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[3])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[4])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[5])
		x = Dropout(0.5)(x)
		x = Flatten()(x)
		x = Dense(units=1164)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[6])
		x = Dense(units=100)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[7])
		x = Dense(units=50)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[8])
		x = Dense(units=10)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[9])
		x = Dense(units=output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'MobileNet':
		def MobilNetInitialConvBlock(inputs, filters, kernel=(3, 3), strides=(1, 1)):
			x = add_custom_layers(inputs,aging_layer=True,aging_active = aging_active[0])
			x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
			x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides)(x)
			x = add_custom_layers(x,aging_layer=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
			return x

		def DepthwiseConvBlock(inputs, filters, strides=(1, 1), blockId=1):
			pad = 'same' if strides == (1, 1) else 'valid'
			x = inputs   if strides == (1, 1) else ZeroPadding2D(((0, 1), (0, 1)))(inputs)
			x = DepthwiseConv2D((3, 3), padding=pad, strides=strides, use_bias=False)(x)
			x = add_custom_layers(x,aging_layer=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2*blockId])
			x = Conv2D(filters, (1, 1), padding='same', strides=(1, 1), use_bias=False)(x)
			x = add_custom_layers(x,aging_layer=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2*blockId+1])
			return x

		input_layer = tf.keras.Input(input_shape)
		x = MobilNetInitialConvBlock(input_layer, 32, strides=(2, 2))
		x = DepthwiseConvBlock(x, 64,   blockId=1)
		x = DepthwiseConvBlock(x, 128,  blockId=2, strides=(2, 2))
		x = DepthwiseConvBlock(x, 128,  blockId=3)
		x = DepthwiseConvBlock(x, 256,  blockId=4, strides=(2, 2))
		x = DepthwiseConvBlock(x, 256,  blockId=5)
		x = DepthwiseConvBlock(x, 512,  blockId=6, strides=(2, 2))
		x = DepthwiseConvBlock(x, 512,  blockId=7)
		x = DepthwiseConvBlock(x, 512,  blockId=8)
		x = DepthwiseConvBlock(x, 512,  blockId=9)
		x = DepthwiseConvBlock(x, 512,  blockId=10)
		x = DepthwiseConvBlock(x, 512,  blockId=11)
		x = DepthwiseConvBlock(x, 1024, blockId=12, strides=(2, 2))
		x = DepthwiseConvBlock(x, 1024, blockId=13)
		x = GlobalAveragePooling2D()(x)
		x = Reshape((1, 1, 1024))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[28])
		x = Dropout(1e-3)(x)
		x = Conv2D(output_shape, (1, 1), padding='same')(x)
		x = add_custom_layers(x,aging_layer=False)
		x = Reshape((output_shape,))(x)
		x = Activation(activation='softmax')(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ZFNet':

		input_layer = tf.keras.Input(input_shape)
		x = add_custom_layers(input_layer,aging_layer=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), padding = 'valid')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2])
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding = 'same')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[4])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[5])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[6])
		x = Conv2D(filters = 256, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[8],quantization_layer=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[9])
		x = Dense(4096)(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[10])
		x = Dense(output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = Activation(activation='softmax')(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SqueezeNet':

		def FireBlock(inputs, fs, fe, blockId):
			#Squeeze
			s  = Conv2D(fs, 1)(inputs)
			s  = ReLU()(s)
			s  = add_custom_layers(s,aging_layer=True,aging_active = aging_active[blockId])
			#Expand
			e1 = Conv2D(fe, 1)(s)
			e1 = ReLU()(e1)
			e3 = Conv2D(fe, 3, padding = 'same')(s)
			e3 = ReLU()(e3)
			e  = Concatenate()([e1,e3])
			e = add_custom_layers(e,aging_layer=True,aging_active = aging_active[blockId+1])
			return e

		input_layer = tf.keras.Input(input_shape)
		x = add_custom_layers(input_layer,aging_layer=True,aging_active = aging_active[0])
		x = Conv2D(96,7,2,'same')(x)
		x = ReLU()(x) 
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = MaxPool2D(3,2,'same')(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2], quantization_layer=False)
		x = FireBlock(x, 16, 64,  3)
		x = FireBlock(x, 16, 64,  5)
		x = FireBlock(x, 32, 128, 7)
		x = MaxPool2D(3,2,'same')(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[9], quantization_layer=False)
		x = FireBlock(x, 32, 128, 10)
		x = FireBlock(x, 48, 192, 12)
		x = FireBlock(x, 48, 192, 14)
		x = FireBlock(x, 64, 256, 16)
		x = MaxPool2D(3,2,'same')(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[18], quantization_layer=False)
		x = FireBlock(x, 64, 256, 19)
		x = Conv2D(output_shape,1,(1,1),'same')(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[20])
		x = GlobalAveragePooling2D()(x)
		x = add_custom_layers(x,aging_layer=False)
		x = tf.keras.activations.softmax(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SentimentalNet':
		top_words = 5000
		max_words = 500	

		input_layer = tf.keras.Input(input_shape)
		x = Embedding(top_words, 32, input_length=max_words)(input_layer)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[0])
		x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = MaxPooling1D(pool_size=2)(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[2])
		x = Flatten()(x)
		x = Dense(250)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[3])
		x = Dense(output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = tf.keras.activations.sigmoid(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'DenseNet':

		def ConvBlock(inputs, growthRate, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(inputs)
			x = ReLU()(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[blockId])
			x = Conv2D(4*growthRate, 1, use_bias=False)(x)
			x = add_custom_layers(x,aging_layer=False)
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[blockId+1])
			x = Conv2D(growthRate, 3, padding='same', use_bias=False)(x)
			x = add_custom_layers(x,aging_layer=False)
			o = Concatenate()([inputs, x])
			return o

		def DenseBlock(x, blocks, blockId):
			for i in range(blocks):
				x = ConvBlock(x, 32, blockId+2*i)
			return x

		def TransitionBlock(x, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[blockId])
			x = Conv2D(int(K.int_shape(x)[-1] * 0.5), 1, use_bias=False)(x)
			x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[blockId+1])
			x = AveragePooling2D(2, strides=2)(x)
			x = add_custom_layers(x,aging_layer=False)
			return x

		input_layer = tf.keras.Input(input_shape)
		x = add_custom_layers(input_layer,aging_layer=True,aging_active = aging_active[0])
		x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
		x = Conv2D(64, 7, strides = 2, use_bias = False)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[1])
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
		x = MaxPool2D(3, strides=2)(x)
		x = DenseBlock(x, 6, 2)
		x = TransitionBlock(x, 14)
		x = DenseBlock(x, 12, 16)
		x = TransitionBlock(x, 40)
		x = DenseBlock(x, 24,  42)
		x = TransitionBlock(x, 90)
		x = DenseBlock(x, 16,  92)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[124])
		x = GlobalAveragePooling2D()(x)
		x = add_custom_layers(x,aging_layer=True,aging_active = aging_active[125])
		x = Dense(output_shape)(x)
		x = add_custom_layers(x,aging_layer=False)
		x = tf.keras.activations.softmax(x)
		x = add_custom_layers(x,aging_layer=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net
