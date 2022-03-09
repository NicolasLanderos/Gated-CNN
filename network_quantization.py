import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import tensorflow as tf
from Simulation import load_obj
from Simulation import save_obj
from Training import get_datasets
from Nets  import get_neural_network_model
from Stats import quantization_effect

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Analisis of quantization effect on a given trained network/dataset')
	parser.add_argument('--batch', metavar='bs', type=int, default = 32,
                         help='batch size to use during inference')
	parser.add_argument('--dataset', metavar='ds', type=str, required = True,
                         help='dataset from https://www.tensorflow.org/datasets/catalog/overview')
	parser.add_argument('--network_name', metavar='nn', type=str, required = True, choices = ['AlexNet','ZFNet','SqueezeNet','MobileNet','VGG16','DenseNet','PilotNet','SentimentalNet'],
                         help='network architecture to train')
	parser.add_argument('--base_bits', metavar='bb', type=int, default = 16,
                         help='number of bits used as base for fractional and integer part')
	args = parser.parse_args()
	tf.random.set_seed(1234)
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, args.network_name)
	wgt_dir = os.path.join(wgt_dir, args.dataset)
	train_info = load_obj(wgt_dir+'/Training_info')

	_,_,test_set,_,_,test_size = get_datasets(args.dataset,train_info.distribution,train_info.input_shape[0:2], train_info.output_shape, train_info.batch, args.batch, return_size = True)
	Net = get_neural_network_model(args.network_name,train_info.input_shape,train_info.output_shape, quantization = False, aging_active=False)
	loss      = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	Net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	Net.load_weights(wgt_dir+'/Weights')
	(OrigLoss,OrigAcc) = Net.evaluate(test_set,verbose = 0)
	print('Non quantized Accuracy over test set:', 'accuracy = ',OrigAcc, 'loss = ', OrigLoss)
	df = quantization_effect(args.network_name,test_set,wgt_dir+'/Weights',train_info.input_shape,train_info.output_shape,args.batch,args.base_bits)
	save_obj(df,wgt_dir+'/Quantization_resume')