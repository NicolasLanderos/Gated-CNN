import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
import pandas as pd
from Simulation import load_obj
from Simulation import save_obj
from Nets  import get_neural_network_model
from Stats import get_read_and_writes

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Elaboration of read/write stats on the buffer, activation size = address size = 16 bits')
	parser.add_argument('--dataset', metavar='ds', type=str, required = True,
                         help='dataset from https://www.tensorflow.org/datasets/catalog/overview')
	parser.add_argument('--network_name', metavar='nn', type=str, required = True, choices = ['AlexNet','ZFNet','SqueezeNet','MobileNet','VGG16','DenseNet','PilotNet','SentimentalNet'],
                         help='network architecture to train')
	parser.add_argument('--addressing_space', metavar='an', type=int, default = 1024*1024,
                         help='number of address in the buffer')
	parser.add_argument('--samples', metavar='sp', type=int, required = True,
                         help='number of samples')
	args = parser.parse_args()
	tf.random.set_seed(1234)
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, args.network_name)
	wgt_dir = os.path.join(wgt_dir, args.dataset)
	train_info = load_obj(wgt_dir+'/Training_info')
	Net = get_neural_network_model(args.network_name,train_info.input_shape,train_info.output_shape, quantization = False, aging_active=False)
	loss      = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	Net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	if args.network_name == 'AlexNet':
		Indices = [0,3,9,11,17,19,25,31,37,40,45,50]
	elif args.network_name == 'ZFNet':
		Indices = [0,3,7,11,15,19,23,27,31,34,37,40]
	elif args.network_name == 'SqueezeNet':
		Indices = [0,3,7, 9,(13,14),20,(24,25),31,(35,36),42,44,(48,49),55,(59,60),66,(70,71),77,(81,82),88,90,(94,95),101,104]
	elif args.network_name == 'MobileNet':
		Indices = [0,4,10,16,23,29,35,41,48,54,60,66,73,79,85,91,97,103,109,115,121,127,133,139,146,152,158,164,170,175]
	elif args.network_name == 'VGG16':
		Indices = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
	elif args.network_name == 'DenseNet':
		Indices = [0,4,11,12,16,(22,11),25,29,(35,24),38,42,(48,37),51,55,(61,50),64,68,(74,63),77,81,(87,76),90,94,97,99,103,(109,97),
        112,116,(122,111),125,129,(135,124),138,142,(148,137),151,155,(161,150),164,168,(174,163),177,181,(187,176),
        190,194,(200,189),203,207,(213,202),216,220,(226,215),229,233,(239,228),242,246,(252,241),255,259,262,264,268,(274,262),
        277,281,(287,276),290,294,(300,289),303,307,(313,302),316,320,(326,315),329,333,(339,328),342,346,(352,341),
        355,359,(365,354),368,372,(378,367),381,385,(391,380),394,398,(404,393),407,411,(417,406),420,424,(430,419),
        433,437,(443,432),446,450,(456,445),459,463,(469,458),472,476,(482,471),485,489,(495,484),498,502,(508,497),
        511,515,(521,510),524,528,(534,523),537,541,(547,536),550,554,(560,549),563,567,(573,562),576,580,583,585,589,(595,583),
        598,602,(608,597),611,615,(621,610),624,628,(634,623),637,641,(647,636),650,654,(660,649),663,667,(673,662),
        676,680,(686,675),689,693,(699,688),702,706,(712,701),715,719,(725,714),728,732,(738,727),741,745,(751,740),
        754,758,(764,753),767,771,(777,765),780,784,(790,779),793,797,800]
	elif args.network_name == 'PilotNet':
		Indices = [5,6,10,14,18,22,28,32,36,40,44]
	elif args.network_name == 'SentimentalNet':
		Indices  = [1,4,8,12,16]
	# Without CNN_gating
	Data     = get_read_and_writes(Net,Indices,args.addressing_space,args.samples,CNN_gating=False)
	stats    = {'Read': Data['Reads'],'Writes': Data['Writes']}
	Baseline = pd.DataFrame(stats).reset_index(drop=False)
	# With CNN_gating
	Data     = get_read_and_writes(Net,Indices,args.addressing_space,args.samples,CNN_gating=True)
	stats    = {'Read': Data['Reads'],'Writes': Data['Writes']}
	CNN_gating = pd.DataFrame(stats).reset_index(drop=False)
	save_obj(Baseline,wgt_dir + '/Baseline_access')
	save_obj(CNN_gating,wgt_dir + '/CNN_gating_access')