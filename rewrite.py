from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random, time

class CNN_AUTOENCODER():
	def __init__(self, shape, learning_rate=0.5, weights=None, biases=None):
		self.shape = shape;
		self.learning_rate = learning_rate
		self.weights = weights
		self.biases = biases
		self.graph = tf.Graph()
		with self.graph.as_default():
			### Input
			self.train			 = tf.placeholder(tf.float32, shape=[None]+self.shape)
			self.train_targets	 = tf.placeholder(tf.float32, shape=[None]+self.shape)
			self.test			 = tf.placeholder(tf.float32, shape=[None]+self.shape)
			self.test_targets	 = tf.placeholder(tf.float32, shape=[None]+self.shape)
			
			### Weights and Biases
			if not self.weights and not self.biases:
				self.weights = {
					'conv1': tf.Variable(tf.truncated_normal(shape=(5,5,1,8), stddev=0.1)),
					'conv3': tf.Variable(tf.truncated_normal(shape=(5,5,8,16), stddev=0.1)),
					'conv5': tf.Variable(tf.truncated_normal(shape=(5,5,16,8), stddev=0.1)),
					'conv7': tf.Variable(tf.truncated_normal(shape=(5,5,8,1), stddev=0.1)),
					'conv9': tf.Variable(tf.truncated_normal(shape=(5,5,1,1),  stddev=0.1)),		  
				}
				self.biases = {
					'conv1': tf.Variable(tf.zeros( shape=(8) )),
					'conv3': tf.Variable(tf.zeros( shape=(16) )),
					'conv5': tf.Variable(tf.zeros( shape=(8) )),
					'conv7': tf.Variable(tf.zeros( shape=(1) )),
					'conv9': tf.Variable(tf.zeros( shape=(1) )),
				}
			
			### Model
			conv1 = self.getConv2DLayer(self.train,
										self.weights['conv1'], self.biases['conv1'],
										activation=tf.nn.relu)
										
			pool2 = tf.nn.max_pool(conv1,
								   ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
								  
			conv3 = self.getConv2DLayer(pool2,
										self.weights['conv3'],self.biases['conv3'],
										activation=tf.nn.relu)
									  
			self.encoded = tf.nn.max_pool(conv3,
								   ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
			
			conv5 = self.getConv2DLayer(self.encoded,
										self.weights['conv5'], self.biases['conv5'],
										activation=tf.nn.relu)
			
			upsa6 = tf.image.resize_images(conv5, [14,14], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			
			conv7 = self.getConv2DLayer(upsa6,
										self.weights['conv7'], self.biases['conv7'],
										activation=tf.nn.relu)
			
			upsa8 = tf.image.resize_images(conv7, [28,28], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			
			self.decoded = self.getConv2DLayer(upsa8,
										self.weights['conv9'], self.biases['conv9'],
										activation=tf.nn.relu)
										
			### Optimizer
			self.loss = tf.losses.mean_squared_error(self.decoded, self.train_targets)
			optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
			self.train_op = optimizer.minimize(self.loss)
			
			### Initialization
			self.init_op = tf.global_variables_initializer()
			
		self.sess = tf.Session(graph=self.graph)
	
	def getConv2DLayer(self, input, weight, bias,
					   strides=(1,1), padding='SAME', activation=None):
		x = tf.add(
			  tf.nn.conv2d(input,
						   weight,
						   [1,strides[0],strides[1],1],
						   padding=padding),
			  bias)
		if activation:
			x = activation(x)
		return x
	
	def fit(self, X, Y, test=None, test_target=None, epochs=5, batch_size=32, suffle = 0):
		if suffle:
			np.random.shuffle(X)
		N = X.shape[0]
		batch_num = int(N / batch_size)
		self.sess.run(self.init_op)
		for epoch in range(epochs):
			for i in range(batch_num):
				batch_train = X[i*batch_size : min((i + 1) * batch_size, N)]
				batch_target = Y[i*batch_size : min((i + 1) * batch_size, N)]
				feed_dict = {
					self.train: batch_train, 
					self.train_targets: batch_target, 
				}
				_, loss, encoded, decoded = self.sess.run([self.train_op, self.loss, self.encoded, self.decoded], feed_dict)
				print("Epoch %2d/%2d:  [%d/%d] loss = %.4f	   "%(epoch+1,epochs,i * batch_size,N,loss),end='\r')
			print("Epoch %2d/%2d:  loss = %.4f	   "%(epoch+1,epochs,loss))
			self.save(epoch, batch_train, decoded)
				
	def save(self, epoch, train, decoded, save_num = 10):
		for i in range(save_num):
			fig = plt.figure()
			plt.imshow(decoded[i, :, :, 0])
			plt.savefig('train_save\\decoded-' + str(i) + '-' + str(epoch) + '.png')
			plt.cla()
			plt.clf()
			plt.close()
			plt.imshow(train[i, :, :, 0])
			plt.savefig('train_save\\train-' + str(i) + '.png')
			plt.cla()
			plt.clf()
			plt.close()
	
		
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

model_1 = CNN_AUTOENCODER(shape=[28, 28, 1], learning_rate= 0.05,)

model_1.fit(X=x_train,
			Y=x_train,
			epochs=5,
			batch_size = 16,
		   )