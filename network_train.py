import argparse
import tensorflow as tf
import numpy as np
import os
from Training import get_datasets
from Simulation import save_obj
from Nets  import get_neural_network_model

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Training of the desired Neural network')
	parser.add_argument('--batch', metavar='bs', type=int, default = 32,
                         help='batch size to use during training')
	parser.add_argument('--dataset', metavar='ds', type=str, required = True,
                         help='dataset from https://www.tensorflow.org/datasets/catalog/overview')
	parser.add_argument('--distribution', metavar='dtb', type=int, nargs = 3, default = [80,5,15],
                         help='distribution of training/validation/testing')
	parser.add_argument('--input_shape', metavar='is', type=int, nargs = 3, required = True,
                         help='input dimensions (height,width,channels)')
	parser.add_argument('--output_shape', metavar='os', type=int, required = True,
                         help='output dimensions')
	parser.add_argument('--network_name', metavar='nn', type=str, required = True, choices = ['AlexNet','ZFNet','SqueezeNet','MobileNet','VGG16','DenseNet','PilotNet','SentimentalNet'],
                         help='network architecture to train')
	args = parser.parse_args()
	tf.random.set_seed(1234)
	train_set,valid_set,_,train_size,valid_size,_ = get_datasets(args.dataset,args.distribution,args.input_shape[0:2], args.output_shape, args.batch, args.batch, return_size = True)
	Net = get_neural_network_model(args.network_name,args.input_shape,args.output_shape, quantization = False, aging_active=False)
	loss      = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	Net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
	Net.fit(x=train_set,epochs=100,
            steps_per_epoch  =int(np.ceil(train_size / args.batch)),
            validation_data  =valid_set,
            validation_steps =int(np.ceil(valid_size/ args.batch)), 
            callbacks=[earlyStop])
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, args.network_name)
	wgt_dir = os.path.join(wgt_dir, args.dataset)
	Net.save_weights(wgt_dir+'/Weights')
	save_obj(args,wgt_dir+'/Training_info')