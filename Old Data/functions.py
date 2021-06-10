import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from fxpmath import Fxp
import pickle

#Funciones para modificar el DataSet
#----------------------------------------------------------------------------------------------------------------------------------------
# 1-hot encoding
def to_categorical(x_, y_):
    return x_, tf.one_hot(y_, depth=10)

# Resizing 
def AlexNet_resize(image, label):
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

def AlexNet_resize_v2(image):
    image = tf.image.resize(image, (227,227))
    return image

def VGG_resize(image, label):
    # Resize images from 32x32 to 224x224
    image = tf.image.resize(image, (224,224))
    return image, label

def VGG_resize_v2(image):
    image = tf.image.resize(image, (224,224))
    return image


#Funcion para obtener todas las salidas de cada capa
#----------------------------------------------------------------------------------------------------------------------------------------
def get_all_outputs(model, input_data, learning_phase=False):
    outputs = [layer.output for layer in model.layers] # exclude Input
    layers_fn = K.function([model.input, K.symbolic_learning_phase()], outputs)
    return layers_fn([input_data, learning_phase])

#Funcion para cargar imagenes en buffer
#----------------------------------------------------------------------------------------------------------------------------------------
# la imagen es guardada pixel a pixel canal a canal, izquierda a derecha, de arriba a abajo.
def Load_Image(buffer, data, Word_bits, Frac_bits):
    index = 0
    for row in data:
        for column in row:
            for color_pixel in column:
                binary_pixel = Fxp(color_pixel,True,Word_bits,Frac_bits)
                binary_pixel = binary_pixel.bin()
                for bit in binary_pixel:
                    buffer[index] = int(bit)
                    index = index + 1

#Funcion para escribir en buffers
#----------------------------------------------------------------------------------------------------------------------------------------
def write_conv_output(data, offset , buffer, Word_bits, Frac_bits):
    binary_data = Fxp(data,True,Word_bits,Frac_bits).bin()
    array_bits  = np.array(list(''.join(binary_data)),dtype=int)
    buffer[offset:offset + len(array_bits)] = array_bits

#Funcion para actualizar estadisticas de los buffers
#----------------------------------------------------------------------------------------------------------------------------------------
def buffer_stadistics(data_buffer,stats_buffer,n_cycles):
    stats_buffer['cambios_logicos'][stats_buffer['ultimo_valor'] != data_buffer ] += 1
    stats_buffer['ciclos_1'][stats_buffer['ultimo_valor'] == 1] += n_cycles
    stats_buffer['ultimo_valor'] = data_buffer.copy()

#Version optimizada
def buffer_stadistics_opt(new_data,offset,stats_buffer,n_cycles):
    stats_buffer['cambios_logicos'][offset:offset+len(new_data)] += stats_buffer['ultimo_valor'][offset:offset+len(new_data)] != new_data
    stats_buffer['ciclos_1'][stats_buffer['ultimo_valor'] == 1] += n_cycles
    stats_buffer['ultimo_valor'][offset:offset+len(new_data)] = new_data

#Funcion para guardar y cargar objetos
#----------------------------------------------------------------------------------------------------------------------------------------
# Escribir objeto en el subdirectorio obj
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
# Cargar objeto desde el subdirectorio obj
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

