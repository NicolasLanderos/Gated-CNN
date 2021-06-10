import tensorflow as tf
import numpy as np

#train,validation,test: a tuple of arrays for input,label or just a array of inputs or None
#Nclasses: Number of classes
#Bs: Batch size
#shape: desired shape of data
#include_label: set if label is included for train,valid and testing
def create_iterators(train,validation,test,Nclasses,shape,T_Bs = 1,I_Bs = 1,include_label=True):

	if include_label:
		def to_categorical(x_, y_):
			return x_, tf.one_hot(y_, depth = Nclasses)
		def normalize(x_, y_):
			return tf.cast(x_, tf.float32) / 255., y_
		def resize(image, label):
			image = tf.image.resize(image, (shape[0],shape[1]))
			return image, label
		if train:
			train_size = train[0].shape[0]
	else:
		def normalize(x_):
			return tf.cast(x_, tf.float32)
		def resize(image):
			image = tf.image.resize(image, (shape[0],shape[1]))
			return image
		if train:
			train_size = train.shape[0]
	train_dataset = valid_dataset = test_dataset = None

	if train:
		train_dataset = tf.data.Dataset.from_tensor_slices(train)
		train_dataset = train_dataset.shuffle(buffer_size=train_size)
		train_dataset = train_dataset.map(to_categorical)
		train_dataset = train_dataset.map(normalize)
		train_dataset = train_dataset.map(resize)
		train_dataset = train_dataset.batch(T_Bs)
		train_dataset = train_dataset.repeat()
	if validation:
		valid_dataset = tf.data.Dataset.from_tensor_slices(validation)
		valid_dataset = valid_dataset.map(to_categorical)
		valid_dataset = valid_dataset.map(normalize)
		valid_dataset = valid_dataset.map(resize)
		valid_dataset = valid_dataset.batch(T_Bs)
		valid_dataset = valid_dataset.repeat()
	if test:
		test_dataset = tf.data.Dataset.from_tensor_slices(test)
		test_dataset = test_dataset.map(to_categorical)
		test_dataset = test_dataset.map(normalize)
		test_dataset = test_dataset.map(resize)
		test_dataset = test_dataset.batch(I_Bs)
	return train_dataset,valid_dataset,test_dataset

def create_regression_iterators(train,validation,test,T_Bs = 1,I_Bs = 1,include_label=True):
	if include_label:
		if train:
			train_size = train[0].shape[0]
	else:
		if train:
			train_size = train.shape[0]
	train_dataset = valid_dataset = test_dataset = None
	if train:
		train_dataset = tf.data.Dataset.from_tensor_slices(train)
		train_dataset = train_dataset.shuffle(buffer_size=train_size)
		train_dataset = train_dataset.batch(T_Bs)
		train_dataset = train_dataset.repeat()
	if validation:
		valid_dataset = tf.data.Dataset.from_tensor_slices(validation)
		valid_dataset = valid_dataset.batch(T_Bs)
		valid_dataset = valid_dataset.repeat()
	if test:
		test_dataset = tf.data.Dataset.from_tensor_slices(test)
		test_dataset = test_dataset.batch(I_Bs)
	return train_dataset,valid_dataset,test_dataset