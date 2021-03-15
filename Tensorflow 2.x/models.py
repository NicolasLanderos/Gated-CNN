import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPool2D, Flatten, Dropout, Lambda

# Capa de Cuantizacion
def Quantization_layer(tensor, Quantization = True, signed = True, word_size = 12, frac_size = 6):
    
    factor = 2.0**frac_size
    
    # Quantized max and min values, in case of the need to implement overflow cases.
    if signed:
        Max_Qvalue = ((1 << (word_size-1)) - 1)/factor
        Min_Qvalue = -Max_Qvalue - 1
    else:
        Max_Qvalue = ((1 << (word_size)) - 1)/factor
        Min_Qvalue = 0
    
    output = tf.round(tensor*factor) / factor
    output = tf.math.minimum(output,Max_Qvalue)   # Upper Saturation
    output = tf.math.maximum(output,Min_Qvalue)   # Lower Saturation
    
    if Quantization:
        return output               
    else:
        return tensor                                       #Simple Bypass

# Cuantizacion de pesos
def Weight_Quantization(model, Frac_Bits, Int_Bits, Dense_Frac_Bits = None, Dense_Int_Bits = None):
    if Dense_Frac_Bits == None:
        Dense_Frac_Bits = Frac_Bits
    if Dense_Int_Bits  == None:
        Dense_Int_Bits  = Int_Bits
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:                   # Layer with weights
            # Quantization of Weights and Bias 
            if layer.__class__.__name__ == 'Dense':
                Qweights    = [Quantization_layer(itm, word_size = (Dense_Int_Bits+Dense_Frac_Bits+1), frac_size = Dense_Frac_Bits) for itm in weights]
            else:
                Qweights    = [Quantization_layer(itm, word_size = (Int_Bits+Frac_Bits+1), frac_size = Frac_Bits) for itm in weights]
            layer.set_weights(Qweights)

# Modelo Lenet
def Lenet_body(input_layer, Quantization = True, signed = True, word_size = 12, frac_size = 6 ):
    Arguments = {'Quantization':Quantization, 'signed':signed, 'word_size':word_size, 'frac_size':frac_size}
    QInp      = tf.keras.layers.Lambda(Quantization_layer, name="QInp",  arguments = Arguments )(input_layer)
    #Conv Block
    Conv1   = tf.keras.layers.Conv2D(6, kernel_size=5, strides=1, input_shape=(28,28,1), padding='same', name= 'Conv1')(QInp)
    QConv1  = tf.keras.layers.Lambda(Quantization_layer, name="QConv1",  arguments = Arguments )(Conv1)
    Act1    = tf.keras.activations.tanh(QConv1)
    QAct1   = tf.keras.layers.Lambda(Quantization_layer, name="QAct1",   arguments = Arguments )(Act1)
    AvgPool1= tf.keras.layers.AveragePooling2D(name='AvgPool1')(QAct1)
    #Conv Block
    Conv2   = tf.keras.layers.Conv2D(16, kernel_size=5, strides=1, padding='valid',name='Conv2')(AvgPool1)
    QConv2  = tf.keras.layers.Lambda(Quantization_layer, name="QConv2",  arguments = Arguments )(Conv2)
    Act2    = tf.keras.activations.tanh(QConv2)
    QAct2   = tf.keras.layers.Lambda(Quantization_layer, name="QAct2",   arguments = Arguments )(Act2)
    AvgPool2= tf.keras.layers.AveragePooling2D(name='AvgPool2')(QAct2)
    Flatten = tf.keras.layers.Flatten(name='Flatten')(AvgPool2)
    #Dense Block
    Dense1  = tf.keras.layers.Dense(units=120, name='Dense1')(Flatten)
    QDense1 = tf.keras.layers.Lambda(Quantization_layer, name="QDense1", arguments = Arguments )(Dense1)
    Act3    = tf.keras.activations.tanh(QDense1)
    QAct3   = tf.keras.layers.Lambda(Quantization_layer, name="QAct3",   arguments = Arguments )(Act3)
    #Dense Block
    Dense2  = tf.keras.layers.Dense(units=84, name='Dense2')(QAct3)
    QDense2 = tf.keras.layers.Lambda(Quantization_layer, name="QDense2", arguments = Arguments)(Dense2)
    Act4    = tf.keras.activations.tanh(QDense2)
    QAct4   = tf.keras.layers.Lambda(Quantization_layer, name="QAct4",   arguments = Arguments)(Act4)
    #Output Block
    Out     = tf.keras.layers.Dense(units=10,name='Output')(QAct4)
    QOut    = tf.keras.layers.Lambda(Quantization_layer, name="QOut",    arguments = Arguments)(Out)
    Act5    = tf.keras.activations.softmax(QOut)
    QAct5   = tf.keras.layers.Lambda(Quantization_layer, name="QSoftmax",arguments = Arguments)(Act5)
    
    return QAct5

# Modelo AlexNet
def AlexNet_body(input_layer, Quantization = True, signed = True, word_size = 12, frac_size = 6 ):
    
    Arguments = {'Quantization':Quantization, 'signed':signed, 'word_size':word_size, 'frac_size':frac_size}
    QInp      = Lambda(Quantization_layer, arguments = Arguments )(input_layer)
    
    #Conv Block
    Conv1   = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4))(QInp)
    QConv1  = Lambda(Quantization_layer, arguments = Arguments )(Conv1)
    Relu1   = tf.keras.activations.relu(QConv1)
    QRelu1  = Lambda(Quantization_layer, arguments = Arguments )(Relu1)
    BN1     = BatchNormalization()(QRelu1)
    QBN1    = Lambda(Quantization_layer, arguments = Arguments )(BN1)
    MP1     = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN1)
    
    Conv2   = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same")(MP1)
    QConv2  = Lambda(Quantization_layer, arguments = Arguments )(Conv2)
    Relu2   = tf.keras.activations.relu(QConv2)
    QRelu2  = Lambda(Quantization_layer, arguments = Arguments )(Relu2)
    BN2     = BatchNormalization()(QRelu2)
    QBN2    = Lambda(Quantization_layer, arguments = Arguments )(BN2)
    MP2     = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN2)
    
    Conv3   = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(MP2)
    QConv3  = Lambda(Quantization_layer, arguments = Arguments )(Conv3)
    Relu3   = tf.keras.activations.relu(QConv3)
    QRelu3  = Lambda(Quantization_layer, arguments = Arguments )(Relu3)
    BN3     = BatchNormalization()(QRelu3)
    QBN3    = Lambda(Quantization_layer, arguments = Arguments )(BN3)
    
    Conv4   = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same")(QBN3)
    QConv4  = Lambda(Quantization_layer, arguments = Arguments )(Conv4)
    Relu4   = tf.keras.activations.relu(QConv4)
    QRelu4  = Lambda(Quantization_layer, arguments = Arguments )(Relu4)
    BN4     = BatchNormalization()(QRelu4)
    QBN4    = Lambda(Quantization_layer, arguments = Arguments )(BN4)
    
    Conv5   = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same")(QBN4)
    QConv5  = Lambda(Quantization_layer, arguments = Arguments )(Conv5)
    Relu5   = tf.keras.activations.relu(QConv5)
    QRelu5  = Lambda(Quantization_layer, arguments = Arguments )(Relu5)
    BN5     = BatchNormalization()(QRelu5)
    QBN5    = Lambda(Quantization_layer, arguments = Arguments )(BN5)
    MP5     = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN5)
    
    Flat    = Flatten()(MP5)
    
    Dense6  = Dense(4096)(Flat)
    QDense6 = Lambda(Quantization_layer, arguments = Arguments )(Dense6)
    Relu6   = tf.keras.activations.relu(QDense6)
    QRelu6  = Lambda(Quantization_layer, arguments = Arguments )(Relu6)
    Drop6   = Dropout(0.5)(QRelu6)
    
    Dense7  = Dense(4096)(Drop6)
    QDense7 = Lambda(Quantization_layer, arguments = Arguments )(Dense7)
    Relu7   = tf.keras.activations.relu(QDense7)
    QRelu7  = Lambda(Quantization_layer, arguments = Arguments )(Relu7)
    Drop7   = Dropout(0.5)(QRelu7)
    
    Dense8  = Dense(10)(Drop7)
    QDense8 = Lambda(Quantization_layer, arguments = Arguments )(Dense8)
    SM8     = tf.keras.activations.softmax(QDense8)
    QSM8    = Lambda(Quantization_layer, arguments = Arguments )(SM8)
    
    return QSM8

# Modelo VGG16
def VGG16_body(input_layer, Quantization = True, signed = True, word_size = 12, frac_size = 6 ):
    
    Arguments = {'Quantization':Quantization, 'signed':signed, 'word_size':word_size, 'frac_size':frac_size}
    QInp      = Lambda(Quantization_layer, arguments = Arguments )(input_layer)
    
    #Conv Block
    Conv1   = Conv2D(filters=64,kernel_size=(3,3),padding="same")(QInp)
    QConv1  = Lambda(Quantization_layer, arguments = Arguments )(Conv1)
    Relu1   = tf.keras.activations.relu(QConv1)
    QRelu1  = Lambda(Quantization_layer, arguments = Arguments )(Relu1)
    
    Conv2   = Conv2D(filters=64,kernel_size=(3,3),padding="same")(QRelu1)
    QConv2  = Lambda(Quantization_layer, arguments = Arguments )(Conv2)
    Relu2   = tf.keras.activations.relu(QConv2)
    QRelu2  = Lambda(Quantization_layer, arguments = Arguments )(Relu2)
    MP2     = MaxPool2D(pool_size=(2,2),strides=(2,2))(QRelu2)
    
    Conv3   = Conv2D(filters=128, kernel_size=(3,3), padding="same")(MP2)
    QConv3  = Lambda(Quantization_layer, arguments = Arguments )(Conv3)
    Relu3   = tf.keras.activations.relu(QConv3)
    QRelu3  = Lambda(Quantization_layer, arguments = Arguments )(Relu3)
    
    Conv4   = Conv2D(filters=128, kernel_size=(3,3), padding="same")(QRelu3)
    QConv4  = Lambda(Quantization_layer, arguments = Arguments )(Conv4)
    Relu4   = tf.keras.activations.relu(QConv4)
    QRelu4  = Lambda(Quantization_layer, arguments = Arguments )(Relu4)
    MP4     = MaxPool2D(pool_size=(2,2),strides=(2,2))(QRelu4)
    
    Conv5   = Conv2D(filters=256, kernel_size=(3,3), padding="same")(MP4)
    QConv5  = Lambda(Quantization_layer, arguments = Arguments )(Conv5)
    Relu5   = tf.keras.activations.relu(QConv5)
    QRelu5  = Lambda(Quantization_layer, arguments = Arguments )(Relu5)
    
    Conv6   = Conv2D(filters=256, kernel_size=(3,3), padding="same")(QRelu5)
    QConv6  = Lambda(Quantization_layer, arguments = Arguments )(Conv6)
    Relu6   = tf.keras.activations.relu(QConv6)
    QRelu6  = Lambda(Quantization_layer, arguments = Arguments )(Relu6)
    
    Conv7   = Conv2D(filters=256, kernel_size=(3,3), padding="same")(QRelu6)
    QConv7  = Lambda(Quantization_layer, arguments = Arguments )(Conv7)
    Relu7   = tf.keras.activations.relu(QConv7)
    QRelu7  = Lambda(Quantization_layer, arguments = Arguments )(Relu7)
    MP7     = MaxPool2D(pool_size=(2,2),strides=(2,2))(QRelu7)
    
    Conv8   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(MP7)
    QConv8  = Lambda(Quantization_layer, arguments = Arguments )(Conv8)
    Relu8   = tf.keras.activations.relu(QConv8)
    QRelu8  = Lambda(Quantization_layer, arguments = Arguments )(Relu8)
    
    Conv9   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(QRelu8)
    QConv9  = Lambda(Quantization_layer, arguments = Arguments )(Conv9)
    Relu9   = tf.keras.activations.relu(QConv9)
    QRelu9  = Lambda(Quantization_layer, arguments = Arguments )(Relu9)
    
    Conv10   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(QRelu9)
    QConv10  = Lambda(Quantization_layer, arguments = Arguments )(Conv10)
    Relu10   = tf.keras.activations.relu(QConv10)
    QRelu10  = Lambda(Quantization_layer, arguments = Arguments )(Relu10)
    MP10     = MaxPool2D(pool_size=(2,2),strides=(2,2))(QRelu10)
    
    Conv11   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(MP10)
    QConv11  = Lambda(Quantization_layer, arguments = Arguments )(Conv11)
    Relu11   = tf.keras.activations.relu(QConv11)
    QRelu11  = Lambda(Quantization_layer, arguments = Arguments )(Relu11)
    
    Conv12   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(QRelu11)
    QConv12  = Lambda(Quantization_layer, arguments = Arguments )(Conv12)
    Relu12   = tf.keras.activations.relu(QConv12)
    QRelu12  = Lambda(Quantization_layer, arguments = Arguments )(Relu12)
    
    Conv13   = Conv2D(filters=512, kernel_size=(3,3), padding="same")(QRelu12)
    QConv13  = Lambda(Quantization_layer, arguments = Arguments )(Conv13)
    Relu13   = tf.keras.activations.relu(QConv13)
    QRelu13  = Lambda(Quantization_layer, arguments = Arguments )(Relu13)
    MP13     = MaxPool2D(pool_size=(2,2),strides=(2,2))(QRelu13)
    
    Flat    = Flatten()(MP13)
    
    Dense14  = Dense(4096)(Flat)
    QDense14 = Lambda(Quantization_layer, arguments = Arguments )(Dense14)
    Relu14   = tf.keras.activations.relu(QDense14)
    QRelu14  = Lambda(Quantization_layer, arguments = Arguments )(Relu14)
    
    Dense15  = Dense(4096)(QRelu14)
    QDense15 = Lambda(Quantization_layer, arguments = Arguments )(Dense15)
    Relu15   = tf.keras.activations.relu(QDense15)
    QRelu15  = Lambda(Quantization_layer, arguments = Arguments )(Relu15)
    
    Dense16  = Dense(10)(QRelu15)
    QDense16 = Lambda(Quantization_layer, arguments = Arguments )(Dense16)
    SM16     = tf.keras.activations.softmax(QDense16)
    QSM16    = Lambda(Quantization_layer, arguments = Arguments )(SM16)
    
    return QSM16