import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import tensorflow as tf
from Simulation import load_obj
from Simulation import save_obj
from Training import get_datasets
from Nets  import get_neural_network_model
from Stats import quantization_effect, weight_quantization
from Simulation import buffer_simulation

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Buffer simulation')
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
	parser.add_argument('--gated_CNN', metavar='CNNg', type=bool, required = True,
                         help='apply or not gated_CNN')
	args = parser.parse_args()
	tf.random.set_seed(1234)
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, args.network_name)
	wgt_dir = os.path.join(wgt_dir, args.dataset)
	train_info = load_obj(wgt_dir+'/Training_info')

	_,_,test_set = get_datasets(args.dataset,train_info.distribution,train_info.input_shape[0:2], train_info.output_shape, train_info.batch, 1)
	Net = get_neural_network_model(args.network_name,train_info.input_shape,train_info.output_shape, quantization = False, aging_active=False)
	loss      = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	Net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
	Net.load_weights(wgt_dir+'/Weights').expect_partial()
	weight_quantization(model = Net, frac_bits = args.wfb, int_bits = args.wib)
	if args.network_name == 'AlexNet':
		LI = [0,3,9,11,17,19,25,31,37,40,45,50]
		AI = [2,8,10,16,18,24,30,36,38,44,49,53]
	elif args.network_name == 'ZFNet':
		LI = [0,3,7 ,11,15,19,23,27,31,34,37,40]
		AI = [2,6,10,14,18,22,26,30,32,36,39,43]
	elif args.network_name == 'SqueezeNet':
		LI = [0,3,7, 9,(13,14),20,(24,25),31,(35,36),42,44,(48,49),55,(59,60),66,(70,71),77,(81,82),88,90,(94,95),101,104]
		AI = [2,6,8,12,     19,23,     30,34,     41,43,47,     54,58,     65,69,     76,80 ,    87,89,93,    100,103,107]
	elif args.network_name == 'MobileNet':
		LI = [0,4,10,16,23,29,35,41,48,54,60,66,73,79,85,91,97,103,109,115,121,127,133,139,146,152,158,164,170,175]
		AI = [2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]
	elif args.network_name == 'VGG16':
		LI = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
		AI = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77]
	elif args.network_name == 'DenseNet':
		LI = [0,4,11,12,16,     22,25,29,     35,38,42,     48,51,55,     61,64,68,     74,77,81,     87,90,94,97,99,103,      109,
     		  112,116,      122,125,129,      135,138,142,      148,151,155,      161,164,168,      174,177,181,      187,
     		  190,194,      200,203,207,      213,216,220,      226,229,233,      239,242,246,      252,255,259,262,264,268,      274,
     		  277,281,      287,290,294,      300,303,307,      313,316,320,      326,329,333,      339,342,346,      352,
     		  355,359,      365,368,372,      378,381,385,      391,394,398,      404,407,411,      417,420,424,      430,
     		  433,437,      443,446,450,      456,459,463,      469,472,476,      482,485,489,      495,498,502,      508,
     		  511,515,      521,524,528,      534,537,541,      547,550,554,      560,563,567,      573,576,580,583,585,589,      595,
     		  598,602,      608,611,615,      621,624,628,      634,637,641,      647,650,654,      660,663,667,      673,
     		  676,680,      686,689,693,      699,702,706,      712,715,719,      725,728,732,      738,741,745,      751,
     		  754,758,      764,767,771,      777,780,784,      790,793,797,800]
		AI = [2,9,11,15,21,(23,11),28,34,(36,24),41,47,(49,37),54,60,(62,50),67,73,(75,63),80,86,(88,76),93,96,98,102,108,(110,98),
     		  115,121,(123,111),128,134,(136,124),141,147,(149,137),154,160,(162,150),167,173,(175,163),180,186,(188,176),
     		  193,199,(201,189),206,212,(214,202),219,225,(227,215),232,238,(240,228),245,251,(253,241),258,261,263,267,273,(275,263),
     		  280,286,(288,276),293,299,(301,289),306,312,(314,302),319,325,(327,315),332,338,(340,328),345,351,(353,341),
     		  358,364,(366,353),371,377,(379,367),384,390,(392,380),397,403,(405,393),410,416,(418,406),423,429,(431,419),
     		  436,442,(444,432),449,455,(457,445),462,468,(470,458),475,481,(483,471),488,494,(496,484),501,507,(509,497),
     		  514,520,(522,510),527,533,(535,523),540,546,(548,536),553,559,(561,549),566,572,(574,562),579,582,584,588,594,(596,584),
     		  601,607,(609,597),614,620,(622,610),627,633,(635,623),640,646,(648,636),653,659,(661,649),666,672,(674,662),
     		  679,685,(687,675),692,698,(700,688),705,711,(713,701),718,724,(726,714),731,737,(739,727),744,750,(752,740),
     		  757,763,(765,753),770,776,(778,766),783,789,(791,779),796,799,803]
	elif args.network_name == 'PilotNet':
		LI = [5,6,10,14,18,22,28,32,36,40,44]
		AI = [5,9,13,17,21,25,31,35,39,43,45]
	elif args.network_name == 'SentimentalNet':
		LI = [1,4,8,12,16]
		AI = [3,7,10,15,19]
	buffer_simulation(Net,test_set, integer_bits = args.aib, fractional_bits = args.afb, samples = args.samples, start_from = 0,
                 bit_invertion = False, bit_shifting = False, CNN_gating = args.gated_CNN, write_mode ='default',
                 results_dir = wgt_dir+'/', buffer_size = 2*args.addressing_space,
                 layer_indexes = LI , activation_indixes = AI)