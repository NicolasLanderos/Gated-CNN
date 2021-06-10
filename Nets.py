import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPool2D, Flatten, Dropout, Lambda, Cropping2D, GlobalAveragePooling2D, Reshape, Activation, ZeroPadding2D, ReLU, DepthwiseConv2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K

def preprocess(image):  # preprocess image
    return tf.image.resize(image, (200, 66))

# Quantizate the inputs with fixed point (sign,magnitude,precision) using saturation arithmetic
def Quantization_layer(tensor, word_size, frac_size, Active = True):
    if Active:
        factor = 2.0**frac_size
    
        Max_Qvalue = ((1 << (word_size-1)) - 1)/factor
        Min_Qvalue = -Max_Qvalue - 1

        output = tf.round(tensor*factor) / factor
        output = tf.math.minimum(output,Max_Qvalue)   # Upper Saturation
        output = tf.math.maximum(output,Min_Qvalue)   # Lower Saturation
        return output               
    else:
        return tensor     

def tf_apply_error(tf_values,errors,word_size,frac_size):
    shift  = 2**(word_size-1)
    factor = 2**frac_size
    tensor = tf.cast(tf_values*factor,dtype=tf.int32)
    tensor = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
    tensor = tf.bitwise.bitwise_and(tensor,errors[:,0])
    tensor = tf.bitwise.bitwise_or(tensor,errors[:,1])
    tensor = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
    tensor = tf.cast(tensor/factor,dtype = tf_values.dtype)
    return tensor

# Implement bit of errors to certain localizations of the layers according to its mapping in the buffer.
def Error_layer(tensor,position_list,error_list,Active,word_size,frac_size):
    if Active:
        values = tf.gather_nd(tensor,position_list)
        return tf.tensor_scatter_nd_update(tensor, position_list, tf_apply_error(values,error_list,word_size,frac_size))
    else:
        return tensor

def decode_mask(mask):
    zero_mask = []
    one_mask  = []
    for item in mask:
        if item == '1':
            one_mask.append('1')
            zero_mask.append('1')
        elif item == '0':
            one_mask.append('0')
            zero_mask.append('0')
        elif item == 'x':
            one_mask.append('0')
            zero_mask.append('1')
    one_error  = int("".join(one_mask),2)
    zero_error = int("".join(zero_mask),2)
    return [zero_error,one_error]

# Generate the tensors of positions and errors to a specific layer plus a flag to indicate if the layer is affected. This objects are used as inputs in Error_layer
# positions: list of positions of the buffer with error
# errors: list of error for the corresponding position in the position list, an example of a error is '1x0' that indicate bit 0 is set to 0, bit 1 is unaffected and bit 2 is set to 1.
# Conv: indicate if the shape of the layer is conv_like or fc_like
def generate_position_list(positions,errors,Conv=False,Shape=(None,None,None),Bs = 1):
    Plist = []
    Elist = []
    is_empty = True
    if Conv:
        for index in range(Bs):
            for position,error in zip(positions,errors):
                if position < Shape[0]*Shape[1]*Shape[2] - 1:
                    act_map  = position//(Shape[0]*Shape[1])
                    row      = (position - act_map*Shape[0]*Shape[1])//Shape[1]
                    col      = position - act_map*Shape[0]*Shape[1] - row*Shape[1]
                    Plist.append([index,row,col,act_map])
                    Elist.append(decode_mask(error))
    else:
        for index in range(Bs):
            for position,error in zip(positions,errors):
                if position < Shape-1:
                    Plist.append([index,position])
                    Elist.append(decode_mask(error))
    if Plist:
        is_empty = False
    return tf.convert_to_tensor(Plist),tf.convert_to_tensor(Elist),not is_empty

#########################################################################################################################################################################################
############################################################################     Models      ############################################################################################
#########################################################################################################################################################################################
# Models are simulated in the follow way:
# 1- Batch Normalization and Activations are Computed in Place.
# 2- Errors layers are only applied to the outputs of layers that must be writed in buffers
# 3- Quantization is not applied to operations that doesnt affect inputs values, like maxpool and Relu

#AlexNet
def AlexNet_body(input_layer, N_labels, locations = [], errors = [], Quantization = True, Errors = True, word_size=None, frac_size=None, Bs = 1):
    if Errors == True:
        Errors = [True]*12
    elif Errors == False:
        Errors = [False]*12

    QArguments = {'Active':Quantization,'word_size':word_size, 'frac_size':frac_size}

    #Input
    QInp       = Lambda(Quantization_layer, arguments = QArguments )(input_layer)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(227,227,3),Bs = Bs)
    QInpError  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[0]),'word_size':word_size, 'frac_size':frac_size})(QInp)
    #Conv Block
    Conv1     = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4))(QInpError)
    QConv1    = Lambda(Quantization_layer, arguments = QArguments )(Conv1)
    Relu1     = tf.keras.activations.relu(QConv1)
    BN1       = BatchNormalization()(Relu1)
    QBN1      = Lambda(Quantization_layer, arguments = QArguments )(BN1)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(55,55,96),Bs = Bs)
    QBN1Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[1]),'word_size':word_size, 'frac_size':frac_size})(QBN1)
    MP1       = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN1Error)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(27,27,96),Bs = Bs)
    MP1Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[2]),'word_size':word_size, 'frac_size':frac_size})(MP1)
    Conv2     = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same")(MP1Error)
    #Conv Block
    QConv2    = Lambda(Quantization_layer, arguments = QArguments )(Conv2)
    Relu2     = tf.keras.activations.relu(QConv2)
    BN2       = BatchNormalization()(Relu2)
    QBN2      = Lambda(Quantization_layer, arguments = QArguments )(BN2)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(27,27,256),Bs = Bs)
    QBN2Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[3]),'word_size':word_size, 'frac_size':frac_size})(QBN2)
    MP2       = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN2Error)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(13,13,256),Bs = Bs)
    MP2Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[4]),'word_size':word_size, 'frac_size':frac_size})(MP2)
    
    Conv3     = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same")(MP2Error)
    QConv3    = Lambda(Quantization_layer, arguments = QArguments )(Conv3)
    Relu3     = tf.keras.activations.relu(QConv3)
    BN3       = BatchNormalization()(Relu3)
    QBN3      = Lambda(Quantization_layer, arguments = QArguments )(BN3)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(13,13,384),Bs = Bs)
    QBN3Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[5]),'word_size':word_size, 'frac_size':frac_size})(QBN3)
    
    Conv4     = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same")(QBN3Error)
    QConv4    = Lambda(Quantization_layer, arguments = QArguments )(Conv4)
    Relu4     = tf.keras.activations.relu(QConv4)
    BN4       = BatchNormalization()(Relu4)
    QBN4      = Lambda(Quantization_layer, arguments = QArguments )(BN4)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(13,13,384),Bs = Bs)
    QBN4Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[6]),'word_size':word_size, 'frac_size':frac_size})(QBN4)
    
    Conv5     = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same")(QBN4Error)
    QConv5    = Lambda(Quantization_layer, arguments = QArguments )(Conv5)
    Relu5     = tf.keras.activations.relu(QConv5)
    BN5       = BatchNormalization()(Relu5)
    QBN5      = Lambda(Quantization_layer, arguments = QArguments )(BN5)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(13,13,256),Bs = Bs)
    QBN5Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[7]),'word_size':word_size, 'frac_size':frac_size})(QBN5)
    MP5       = MaxPool2D(pool_size=(3,3), strides=(2,2))(QBN5Error)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(6,6,256),Bs = Bs)
    MP5Error  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[8]),'word_size':word_size, 'frac_size':frac_size})(MP5)
    
    Flat    = Flatten()(MP5Error)
    
    Dense6  = Dense(4096)(Flat)
    QDense6 = Lambda(Quantization_layer, arguments = QArguments )(Dense6)
    Relu6   = tf.keras.activations.relu(QDense6)
    Drop6   = Dropout(0.5)(Relu6)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=4096,Bs = Bs)
    D6Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[9]),'word_size':word_size, 'frac_size':frac_size})(Drop6)
    
    Dense7  = Dense(4096)(D6Error)
    QDense7 = Lambda(Quantization_layer, arguments = QArguments )(Dense7)
    Relu7   = tf.keras.activations.relu(QDense7)
    Drop7   = Dropout(0.5)(Relu7)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=4096,Bs = Bs)
    D7Error = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[10]),'word_size':word_size, 'frac_size':frac_size})(Drop7)
    
    Dense8  = Dense(N_labels)(D7Error)
    QDense8 = Lambda(Quantization_layer, arguments = QArguments )(Dense8)
    SM8     = tf.keras.activations.softmax(QDense8)
    QSM8    = Lambda(Quantization_layer, arguments = QArguments )(SM8)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=10,Bs = Bs)
    QSError = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[11]),'word_size':word_size, 'frac_size':frac_size})(QSM8)
    
    return QSError


def VGG16_body(input_layer, N_labels, locations = [], errors = [], Quantization = True, Errors = True, word_size=None, frac_size=None, Bs = 1):
    if Errors == True:
        Errors = [True]*21
    elif Errors == False:
        Errors = [False]*21

    QArguments = {'Active':Quantization,'word_size':word_size, 'frac_size':frac_size}

    #Input
    QInp       = Lambda(Quantization_layer, arguments = QArguments )(input_layer)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(224,224,3),Bs = Bs)
    QInpError  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[0]),'word_size':word_size, 'frac_size':frac_size})(QInp)
    #Conv Block
    Conv1      = Conv2D(filters=64,kernel_size=(3,3),padding="same")(QInpError)
    QConv1     = Lambda(Quantization_layer, arguments = QArguments )(Conv1)
    Relu1      = tf.keras.activations.relu(QConv1)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(224,224,64),Bs = Bs)
    Relu1Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[1]),'word_size':word_size, 'frac_size':frac_size})(Relu1)
    #Conv Block
    Conv2      = Conv2D(filters=64,kernel_size=(3,3),padding="same")(Relu1Err)
    QConv2     = Lambda(Quantization_layer, arguments = QArguments )(Conv2)
    Relu2      = tf.keras.activations.relu(QConv2)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(224,224,64),Bs = Bs)
    Relu2Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[2]),'word_size':word_size, 'frac_size':frac_size})(Relu2)
    #Pooling
    MP2        = MaxPool2D(pool_size=(2,2),strides=(2,2))(Relu2Err)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(112,112,64),Bs = Bs)
    MP2Err     = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[3]),'word_size':word_size, 'frac_size':frac_size})(MP2)
    #Conv Block
    Conv3      = Conv2D(filters=128,kernel_size=(3,3),padding="same")(MP2Err)
    QConv3     = Lambda(Quantization_layer, arguments = QArguments )(Conv3)
    Relu3      = tf.keras.activations.relu(QConv3)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(112,112,128),Bs = Bs)
    Relu3Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[4]),'word_size':word_size, 'frac_size':frac_size})(Relu3)
    #Conv Block
    Conv4      = Conv2D(filters=128,kernel_size=(3,3),padding="same")(Relu3Err)
    QConv4     = Lambda(Quantization_layer, arguments = QArguments )(Conv4)
    Relu4      = tf.keras.activations.relu(QConv4)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(112,112,128),Bs = Bs)
    Relu4Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[5]),'word_size':word_size, 'frac_size':frac_size})(Relu4)
    #Pooling
    MP4        = MaxPool2D(pool_size=(2,2),strides=(2,2))(Relu4Err)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(56,56,128),Bs = Bs)
    MP4Err     = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[6]),'word_size':word_size, 'frac_size':frac_size})(MP4)    
    #Conv Block
    Conv5      = Conv2D(filters=256,kernel_size=(3,3),padding="same")(MP4Err)
    QConv5     = Lambda(Quantization_layer, arguments = QArguments )(Conv5)
    Relu5      = tf.keras.activations.relu(QConv5)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(56,56,256),Bs = Bs)
    Relu5Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[7]),'word_size':word_size, 'frac_size':frac_size})(Relu5)
    #Conv Block
    Conv6      = Conv2D(filters=256,kernel_size=(3,3),padding="same")(Relu5Err)
    QConv6     = Lambda(Quantization_layer, arguments = QArguments )(Conv6)
    Relu6      = tf.keras.activations.relu(QConv6)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(56,56,256),Bs = Bs)
    Relu6Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[8]),'word_size':word_size, 'frac_size':frac_size})(Relu6)
    #Conv Block
    Conv7      = Conv2D(filters=256,kernel_size=(3,3),padding="same")(Relu6Err)
    QConv7     = Lambda(Quantization_layer, arguments = QArguments )(Conv7)
    Relu7      = tf.keras.activations.relu(QConv7)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(56,56,256),Bs = Bs)
    Relu7Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[9]),'word_size':word_size, 'frac_size':frac_size})(Relu7)
    #Pooling
    MP7        = MaxPool2D(pool_size=(2,2),strides=(2,2))(Relu7Err)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(28,28,256),Bs = Bs)
    MP7Err     = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[10]),'word_size':word_size, 'frac_size':frac_size})(MP7)    
    #Conv Block
    Conv8      = Conv2D(filters=512,kernel_size=(3,3),padding="same")(MP7Err)
    QConv8     = Lambda(Quantization_layer, arguments = QArguments )(Conv8)
    Relu8      = tf.keras.activations.relu(QConv8)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(28,28,512),Bs = Bs)
    Relu8Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[11]),'word_size':word_size, 'frac_size':frac_size})(Relu8)
    #Conv Block
    Conv9      = Conv2D(filters=512,kernel_size=(3,3),padding="same")(Relu8Err)
    QConv9     = Lambda(Quantization_layer, arguments = QArguments )(Conv9)
    Relu9      = tf.keras.activations.relu(QConv9)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(28,28,512),Bs = Bs)
    Relu9Err   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[12]),'word_size':word_size, 'frac_size':frac_size})(Relu9)
    #Conv Block
    Conv10     = Conv2D(filters=512,kernel_size=(3,3),padding="same")(Relu9Err)
    QConv10    = Lambda(Quantization_layer, arguments = QArguments )(Conv10)
    Relu10     = tf.keras.activations.relu(QConv10)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(28,28,512),Bs = Bs)
    Relu10Err  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[13]),'word_size':word_size, 'frac_size':frac_size})(Relu10)
    #Pooling
    MP10       = MaxPool2D(pool_size=(2,2),strides=(2,2))(Relu10Err)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(14,14,256),Bs = Bs)
    MP10Err    = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[14]),'word_size':word_size, 'frac_size':frac_size})(MP10)    
    #Conv Block
    Conv11     = Conv2D(filters=512,kernel_size=(3,3),padding="same")(MP10Err)
    QConv11    = Lambda(Quantization_layer, arguments = QArguments )(Conv11)
    Relu11     = tf.keras.activations.relu(QConv11)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(14,14,256),Bs = Bs)
    Relu11Err  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[15]),'word_size':word_size, 'frac_size':frac_size})(Relu11)
    #Conv Block
    Conv12     = Conv2D(filters=512,kernel_size=(3,3),padding="same")(Relu11Err)
    QConv12    = Lambda(Quantization_layer, arguments = QArguments )(Conv12)
    Relu12     = tf.keras.activations.relu(QConv12)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(14,14,256),Bs = Bs)
    Relu12Err  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[16]),'word_size':word_size, 'frac_size':frac_size})(Relu12)
    #Conv Block
    Conv13     = Conv2D(filters=512,kernel_size=(3,3),padding="same")(Relu12Err)
    QConv13    = Lambda(Quantization_layer, arguments = QArguments )(Conv13)
    Relu13     = tf.keras.activations.relu(QConv13)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(14,14,256),Bs = Bs)
    Relu13Err  = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[17]),'word_size':word_size, 'frac_size':frac_size})(Relu13)
    #Pooling
    MP13       = MaxPool2D(pool_size=(2,2),strides=(2,2))(Relu13Err)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(7,7,512),Bs = Bs)
    MP13Err    = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[18]),'word_size':word_size, 'frac_size':frac_size})(MP13)    
    #Flattening
    Flat       = Flatten()(MP13Err)
    #FC
    Dense14    = Dense(4096)(Flat)
    QDense14   = Lambda(Quantization_layer, arguments = QArguments )(Dense14)
    Relu14     = tf.keras.activations.relu(QDense14)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=4096,Bs = Bs)
    D14Error   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[19]),'word_size':word_size, 'frac_size':frac_size})(Relu14)
    #FC
    Dense15    = Dense(4096)(D14Error)
    QDense15   = Lambda(Quantization_layer, arguments = QArguments )(Dense15)
    Relu15     = tf.keras.activations.relu(QDense15)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=4096,Bs = Bs)
    D15Error   = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[20]),'word_size':word_size, 'frac_size':frac_size})(Relu15)
    #FC
    Dense16    = Dense(N_labels)(D15Error)
    QDense16   = Lambda(Quantization_layer, arguments = QArguments )(Dense16)
    SM16       = tf.keras.activations.softmax(QDense16)
    QSM16      = Lambda(Quantization_layer, arguments = QArguments )(SM16)

    return QSM16


def PilotNet_model(locations = [], errors = [], Quantization = True, Errors = True, word_size=None, frac_size=None, Bs = 1):
    if Errors == True:
        Errors = [True]*10
    elif Errors == False:
        Errors = [False]*10

    QArguments = {'Active':Quantization,'word_size':word_size, 'frac_size':frac_size}

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(preprocess))
    model.add(Lambda(lambda x: (x/ 127.0 - 1.0)))
    model.add(Lambda(Quantization_layer, arguments = QArguments))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(200,66,3),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[0]),'word_size':word_size, 'frac_size':frac_size}))
    #Conv Block
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(98,31,24),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[1]),'word_size':word_size, 'frac_size':frac_size}))
    #Conv Block
    model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(47,14,36),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[2]),'word_size':word_size, 'frac_size':frac_size}))
    #Conv Block
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(22,5,48),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[3]),'word_size':word_size, 'frac_size':frac_size}))
    # Conv Block
    model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(20,3,64),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[4]),'word_size':word_size, 'frac_size':frac_size}))
    # Conv Block
    model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(18,1,64),Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[5]),'word_size':word_size, 'frac_size':frac_size}))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    #Dense Block
    model.add(Dense(units=1164, activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=1164,Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[6]),'word_size':word_size, 'frac_size':frac_size}))
    #Dense Block
    model.add(Dense(units=100, activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=100,Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[7]),'word_size':word_size, 'frac_size':frac_size}))
    #Dense Block
    model.add(Dense(units=50, activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=50,Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[8]),'word_size':word_size, 'frac_size':frac_size}))
    #Dense Block
    model.add(Dense(units=10, activation='relu'))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Shape=10,Bs = Bs)
    model.add(Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList),'Active': tf.identity(is_Empty and Errors[9]),'word_size':word_size, 'frac_size':frac_size}))
    #Dense Block
    model.add(Dense(units=1))
    model.add(Lambda(Quantization_layer, arguments = QArguments ))

    return model


def MobileNet_body(input_layer, N_labels, locations = [], errors = [], Quantization = True, Errors = True, word_size=None, frac_size=None, Bs = 1):

    Shapes = [(224,224,3),(113,113,32),(113,113,32),(113,113,64),(56,56,64),(56,56,128),(56,56,128),(56,56,128),(28,28,128),(28,28,256),(28,28,256),(28,28,256),(14,14,256),(14,14,512),
              (14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(14,14,512),(7,7,512),(7,7,1024),(7,7,1024),(7,7,1024)]

    def MobilNet_Initial_conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qinput')(input_layer)
        PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(224,224,3),Bs = Bs)
        x = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[0]),
            'word_size':word_size, 'frac_size':frac_size}, name='Qinput_error')(x)

        x = ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(x)
        x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv1')(x)
        x = BatchNormalization(name='conv1_bn')(x)
        x = ReLU(6., name='conv1_relu')(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv1_relu')(x)
        PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(113,113,32),Bs = Bs)
        return Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[1]),
            'word_size':word_size, 'frac_size':frac_size}, name='Qconv1_relu_error')(x)

    def _depthwise_conv_block(inputs, filters, strides=(1, 1), block_id=1):
        if strides == (1, 1):
            x = inputs
            pad = 'same'
        else:
            x = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
            pad = 'valid'
        x = DepthwiseConv2D((3, 3), padding=pad, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv_dw_%d' % block_id)(x)
        x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
        x = ReLU(6., name='conv_dw_%d_relu' % block_id)(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv_dw_%d_relu' % block_id)(x)
        PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=Shapes[2*block_id],Bs = Bs)
        x = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[2*block_id]),
            'word_size':word_size, 'frac_size':frac_size},name = 'Qconv_dw_%d_relu_error' % block_id)(x)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv_pw_%d' % block_id)(x)
        x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
        x = ReLU(6., name='conv_pw_%d_relu' % block_id)(x)
        x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv_pw_%d_relu' % block_id)(x)
        PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=Shapes[2*block_id+1],Bs = Bs)
        return Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[1]),
            'word_size':word_size, 'frac_size':frac_size}, name='Qconv_pw_%d_relu_error' % block_id)(x)

    if Errors == True:
        Errors = [True]*29
    elif Errors == False:
        Errors = [False]*29

    QArguments = {'Active':Quantization,'word_size':word_size, 'frac_size':frac_size}

    x = MobilNet_Initial_conv_block(input_layer, 32, strides=(2, 2))
    x = _depthwise_conv_block(x, 64,   block_id=1)
    
    x = _depthwise_conv_block(x, 128,  block_id=2, strides=(2, 2))
    x = _depthwise_conv_block(x, 128,  block_id=3)

    x = _depthwise_conv_block(x, 256,  block_id=4, strides=(2, 2))
    x = _depthwise_conv_block(x, 256,  block_id=5)

    x = _depthwise_conv_block(x, 512,  block_id=6, strides=(2, 2))
    x = _depthwise_conv_block(x, 512,  block_id=7)
    x = _depthwise_conv_block(x, 512,  block_id=8)
    x = _depthwise_conv_block(x, 512,  block_id=9)
    x = _depthwise_conv_block(x, 512,  block_id=10)
    x = _depthwise_conv_block(x, 512,  block_id=11)

    x = _depthwise_conv_block(x, 1024, block_id=12, strides=(2, 2))
    x = _depthwise_conv_block(x, 1024, block_id=13)
    
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Lambda(Quantization_layer, arguments = QArguments , name='QGAP')(x)
    PosList,ErrList,is_Empty = generate_position_list(locations,errors,Conv=True,Shape=(1,1,1024),Bs = Bs)
    x = Lambda(Error_layer, arguments = {'position_list' : tf.identity(PosList),'error_list' : tf.identity(ErrList), 'Active': tf.identity(is_Empty and Errors[28]),
            'word_size':word_size, 'frac_size':frac_size},name = 'QGAP_error')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(N_labels, (1, 1), padding='same', name='conv_preds')(x)
    x = Lambda(Quantization_layer, arguments = QArguments , name='Qconv_preds')(x)
    x = Reshape((N_labels,), name='reshape_2')(x)
    x = Activation(activation='softmax', name='predictions')(x)
    x = Lambda(Quantization_layer, arguments = QArguments , name='Qpredictions')(x)
    return x