import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import gc
from Nets import Quantization_layer, AlexNet_body, VGG16_body, PilotNet_model, MobileNet_body


def Weight_Quantization(model, Frac_Bits, Int_Bits):
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:                   # Layer with weights
            Qweights    = [Quantization_layer(itm, word_size = (1+Int_Bits+Frac_Bits), frac_size=Frac_Bits) for itm in weights]
            layer.set_weights(Qweights)




def Check_Accuracy_and_Loss(model, test_dataset, Wgt_dir, Afrac_size, Aint_size, Wfrac_size, Wint_size, N_labels, IShape, locations = [], errors = [], Layer_Error_mask = [False]*12, Bs = 1, verbose = 1):

    Qinput_layer  = tf.keras.Input(IShape)
    if model == 'Alex':
        Qoutput_layer = AlexNet_body(Qinput_layer, Quantization = True, N_labels = N_labels, word_size = (1+Aint_size+Afrac_size), frac_size = Afrac_size,
                                    Errors = Layer_Error_mask, locations = locations, errors = errors, Bs = Bs)
    elif model == 'VGG16':
    	Qoutput_layer = VGG16_body(Qinput_layer, Quantization = True, N_labels = N_labels, word_size = (1+Aint_size+Afrac_size), frac_size = Afrac_size,
                                    Errors = Layer_Error_mask, locations = locations, errors = errors, Bs = Bs)
    elif model == 'Mobile':
    	Qoutput_layer = MobileNet_body(Qinput_layer, Quantization = True, N_labels = N_labels, word_size = (1+Aint_size+Afrac_size), frac_size = Afrac_size,
                                    Errors = Layer_Error_mask, locations = locations, errors = errors, Bs = Bs)
    else:
        return None
    Qnet = tf.keras.Model(inputs=Qinput_layer, outputs=Qoutput_layer)
    #Load Weights
    Qnet.load_weights(Wgt_dir).expect_partial()
    #Quantize Weights
    Weight_Quantization(model = Qnet, Frac_Bits = Wfrac_size, Int_Bits = Wint_size)
    # Params
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    metrics = ['accuracy']
    # Compile Model
    Qnet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    (loss,acc) = Qnet.evaluate(test_dataset,verbose=verbose)
    # Cleaning Memory
    del Qnet
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    return acc,loss


def Check_Regression_Loss(model, test_dataset, Wgt_dir, Afrac_size, Aint_size, Wfrac_size, Wint_size, locations = [], errors = [], Layer_Error_mask = [False]*12, Bs = 1, verbose = 1):
	if model == 'Pilot':
		Qnet =  PilotNet_model(locations = locations, errors = errors, Quantization = True, Errors = Layer_Error_mask, word_size=(1+Aint_size+Afrac_size), frac_size=Afrac_size, Bs = Bs)
	else:
		return None
	Qnet.load_weights(Wgt_dir).expect_partial()
	Weight_Quantization(model = Qnet, Frac_Bits = Wfrac_size, Int_Bits = Wint_size)
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	Qnet.compile(optimizer=optimizer, loss='mse')
	loss = Qnet.evaluate(test_dataset,verbose=verbose)
	# Cleaning Memory
	del Qnet
	gc.collect()
	K.clear_session()
	tf.compat.v1.reset_default_graph()
	return loss
