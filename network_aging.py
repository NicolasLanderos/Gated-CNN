import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
import numpy as np
from Simulation import load_obj
from Simulation import save_obj
from Training import get_datasets
from Stats import weight_quantization
from Stats import check_accuracy_and_loss
from datetime import datetime
import itertools

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Aging Simulation')
	parser.add_argument('--dataset', metavar='ds', type=str, required = True,
						 help='dataset from https://www.tensorflow.org/datasets/catalog/overview')
	parser.add_argument('--network_name', metavar='nn', type=str, required = True, choices = ['AlexNet','ZFNet','SqueezeNet','MobileNet','VGG16','DenseNet','PilotNet','SentimentalNet'],
						 help='network architecture to train')
	parser.add_argument('--addressing_space', metavar='an', type=int, default = 1024*1024,
						 help='number of address in the buffer')
	parser.add_argument('--samples', metavar='sp', type=int, required = True,
						 help='number of samples')
	parser.add_argument('--afb', metavar='afb', type=int, required = True,
						 help='activation fractional part number of bits')
	parser.add_argument('--aib', metavar='afb', type=int, required = True,
						 help='activation integer part number of bits')
	parser.add_argument('--wfb', metavar='afb', type=int, required = True,
						 help='weight fractional part number of bits')
	parser.add_argument('--wib', metavar='afb', type=int, required = True,
						 help='weight integer part number of bits')
	parser.add_argument('--wgt_faults', metavar='wgtf', type=bool, required = True,
						 help='True for faults in weight buffer, False for faults in activatin buffer')
	parser.add_argument('--batch', metavar='bs', type=int, required = True,
						 help='batch size of inference')
	parser.add_argument('--portion', metavar='port', type=float, required = True,
						 help='portion of buffer with faults')
	args = parser.parse_args()
	tf.random.set_seed(1234)
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, args.network_name)
	wgt_dir = os.path.join(wgt_dir, args.dataset)
	train_info = load_obj(wgt_dir+'/Training_info')

	_,_,test_set = get_datasets(args.dataset,train_info.distribution,train_info.input_shape[0:2], train_info.output_shape, train_info.batch, args.batch)
	Accs     = {args.portion:[]}
	Loss     = {args.portion:[]}
	
	network_size   = args.addressing_space*16
	num_of_samples = args.samples
	for Enumber in Accs:
		n_bits_fails = np.ceil(Enumber*network_size).astype(int)
		errors       = np.random.randint(0,2,n_bits_fails)
		buffer       = np.array(['x']*(network_size-n_bits_fails))
		buffer       = np.concatenate([buffer,errors])
		for index in range(0,num_of_samples):
			np.random.shuffle(buffer)
			address_with_errors = np.reshape(buffer,(-1,16))
			address_with_errors = enumerate(["".join(i) for i in address_with_errors])
			error_mask = [y for x,y in address_with_errors if y.count('x') < 16]
			locs       = [x for x,y in address_with_errors if y.count('x') < 16]
			del address_with_errors
			loss,acc   = check_accuracy_and_loss(args.network_name, test_set, wgt_dir+'/Weights', output_shape=train_info.output_shape, input_shape = train_info.input_shape,
											act_frac_size = args.afb, act_int_size = args.aib, wgt_frac_size = args.wfb, wgt_int_size = args.wib,
											batch_size=args.batch, verbose = 0, aging_active = not args.wgt_faults, weights_faults = args.wgt_faults,
											faulty_addresses = locs, masked_faults = error_mask)
			Accs[Enumber].append(acc)
			Loss[Enumber].append(loss)
		save_obj(Accs,wgt_dir+'/Accs')
		save_obj(Loss,wgt_dir+'/Loss')