from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import cv2
import os


###################################################################################################
##FUNCTION NAME: get_datasets
##DESCRIPTION:   Prepares the dataset to be used in training/inference
##OUTPUTS:       datasets iterators for training/validation/testing and optionally the sizes
############ARGUMENTS##############################################################################
####dataset_name:     Name of tensorflow datasets
####split:            tuple; partition percentajes for training and validation (rest for testing)
####data_shape:       objetive shape of the images
####target_shape:     output shape
####train_batch_size: desired training batch size
####test_valid_size:  desired validation batch size
####return_size:      True if size of datasets is desired as output too
###################################################################################################

def get_datasets(dataset_name, split, data_shape, target_shape, train_batch_size, test_batch_size,
                return_size=False):
	x_train,y_train= tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train[:'+str(split[0])+'%]'))
	x_valid,y_valid= tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train['+str(split[0])+':'+str(split[0]+split[1])+'%]'))
	x_test, y_test = tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train['+str(split[0]+split[1])+'%:]'))
	def to_categorical(x_, y_):
		return x_, tf.one_hot(y_, depth = target_shape)
	def normalize(x_, y_):
		return tf.cast(x_, tf.float32) / 255., y_
	def resize(image, label):
		image = tf.image.resize(image, (data_shape[0],data_shape[1]))
		return image, label
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
	train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0])
	train_dataset = train_dataset.map(to_categorical)
	train_dataset = train_dataset.map(normalize)
	train_dataset = train_dataset.map(resize)
	train_dataset = train_dataset.batch(train_batch_size)
	train_dataset = train_dataset.repeat()
	valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
	valid_dataset = valid_dataset.map(to_categorical)
	valid_dataset = valid_dataset.map(normalize)
	valid_dataset = valid_dataset.map(resize)
	valid_dataset = valid_dataset.batch(train_batch_size)
	valid_dataset = valid_dataset.repeat()
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_dataset = test_dataset.map(to_categorical)
	test_dataset = test_dataset.map(normalize)
	test_dataset = test_dataset.map(resize)
	test_dataset = test_dataset.batch(test_batch_size)
	if return_size:
		return train_dataset,valid_dataset,test_dataset,x_train.shape[0],x_valid.shape[0],x_test.shape[0]
	else:
		return train_dataset,valid_dataset,test_dataset

###################################################################################################
##FUNCTION NAME: get_IMBD_dataset
##DESCRIPTION:   load imbd dataset for sentimentalNet
##OUTPUTS:       dataset iterators
############ARGUMENTS##############################################################################
####train_batch_size: desired training/validation batch size
####test_valid_size:  desired testing batch size
###################################################################################################

def get_IMBD_dataset(train_batch_size, test_batch_size):
	top_words = 5000
	max_words = 500
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
	x_train = sequence.pad_sequences(x_train, maxlen=max_words)
	x_test  = sequence.pad_sequences(x_test, maxlen=max_words)
	x_valid = x_test[0:10000]
	y_valid = y_test[0:10000]
	x_test  = x_test[10000:]
	y_test  = y_test[10000:]
	Train_Samples = x_train.shape[0]
	Valid_Samples = x_valid.shape[0]
	Test_Samples  = x_test.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
	train_dataset = train_dataset.shuffle(buffer_size=Train_Samples)
	train_dataset = train_dataset.batch(train_batch_size)
	train_dataset = train_dataset.repeat()
	valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
	valid_dataset = valid_dataset.batch(train_batch_size)
	valid_dataset = valid_dataset.repeat()
	test_dataset  = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_dataset  = test_dataset.batch(test_batch_size)
	return train_dataset,valid_dataset,test_dataset