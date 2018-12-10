from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random, time

class AUTOENCODER():
	def __init__(self, x_shape, y_shape, learning_rate=0.05, layers=None, weights=None, activations=None, biases=None):
		self.x_shape = x_shape
		self.y_shape = y_shape
		self.learning_rate = learning_rate
		self.layers = layers
		self.weights = weights
		self.activations = activations
		self.biases = biases
		self.loss_save = []
		self.graph = tf.Graph()
		with self.graph.as_default():
			### Input
			self.train			 = tf.placeholder(tf.float32, shape=[None]+self.x_shape)
			self.train_targets	 = tf.placeholder(tf.float32, shape=[None]+self.y_shape)
			
			### Weights and Biases
			self.check_input()
			
			### Model
			self.encoded, self.decoded = self.build_model(self.train)
			
			### Optimizer
			self.loss = tf.losses.absolute_difference(self.decoded, self.train_targets)
			optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
			self.train_op = optimizer.minimize(self.loss)
			
			### Initialization
			self.init_op = tf.global_variables_initializer()
		
			### Saver
			self.save = tf.train.Saver()
			
			### Evaluate/Predict
			self.new	     = tf.placeholder(tf.float32, shape=[None]+self.x_shape)
			self.new_targets = tf.placeholder(tf.float32, shape=[None]+self.y_shape)
			self.new_encoded, self.new_decoded = self.build_model(self.new)
			self.new_loss = tf.losses.absolute_difference(self.new_decoded, self.new_targets)
			
		self.sess = tf.Session(graph=self.graph)
	
	
	def check_input(self):
		if not self.layers or not self.weights or not self.biases or not self.activations:
			self.layers = [
				'conv1', 'pool2', 'conv3', 'poolEncoded', 
				'conv5', 'upsa6', 'conv7', 'upsa8', 
				'convDecoded',
			]
			self.weights = {
				'conv1': tf.Variable(tf.truncated_normal(shape=(5,5,1,8), stddev=0.1)),
				'pool2': [1, 2, 2, 1],
				'conv3': tf.Variable(tf.truncated_normal(shape=(5,5,8,16), stddev=0.1)),
				'poolEncoded': [1, 2, 2, 1],
				'conv5': tf.Variable(tf.truncated_normal(shape=(5,5,16,8), stddev=0.1)),
				'upsa6': [14, 14],
				'conv7': tf.Variable(tf.truncated_normal(shape=(5,5,8,1), stddev=0.1)),
				'upsa8': [28, 28],
				'convDecoded': tf.Variable(tf.truncated_normal(shape=(5,5,1,1),  stddev=0.1)), 
			}
			self.activations = {
				'conv1': tf.nn.relu,
				'conv3': tf.nn.relu,
				'conv5': tf.nn.relu,
				'conv7': tf.nn.relu,
				'convDecoded': tf.nn.relu,
			}
			self.biases = {
				'conv1': tf.Variable(tf.zeros( shape=(8) )),
				'conv3': tf.Variable(tf.zeros( shape=(16) )),
				'conv5': tf.Variable(tf.zeros( shape=(8) )),
				'conv7': tf.Variable(tf.zeros( shape=(1) )),
				'convDecoded': tf.Variable(tf.zeros( shape=(1) )),
			}
	
	
	def build_model(self, X):
		if self.layers[0][0:4] == 'conv':
			x = self.getConv2DLayer(X,
									self.weights[self.layers[0]], self.biases[self.layers[0]],
									activation=self.activations[self.layers[0]])
		
		for layer in self.layers[1:]:
			if layer[0:4] == 'conv':
				x = self.getConv2DLayer(x,
										self.weights[layer], self.biases[layer],
										activation=self.activations[layer])
										
			elif layer[0:4] == 'pool':
				x = tf.nn.max_pool(x, ksize=self.weights[layer], 
								   strides=self.weights[layer], padding='SAME')
							   
			elif layer[0:4] == 'upsa':
				x = tf.image.resize_images(x, self.weights[layer], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
				# x = tf.keras.layers.UpSampling2D(2)(x)
			
			if layer[4:] == 'Encoded':
				encoded = x
				
			elif layer[4:] == 'Decoded':
				decoded = x
		
		return (encoded, decoded)
	
	
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
	
	
	def fit(self, X, Y, test=None, test_target=None, epochs=5, batch_size=32, suffle=0, eval = 0):
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
				self.loss_save.append(loss)
				print("Epoch %2d/%2d:  [%d/%d] loss = %.4f	   "%(epoch+1, epochs, i*batch_size, N, loss),end='\r')
			
			if eval:
				new_loss = self.evaluate(test, test_target, batch_size)
			
			print("Epoch %2d/%2d:  loss = %.4f  eval_loss = %.4f"%(epoch+1, epochs, loss, new_loss))
			self.save_img(epoch, batch_train, decoded)
	
	
	def evaluate(self, X, Y, batch_size):
		N = X.shape[0]
		batch_num = int(N / batch_size)
		total = 0
		for i in range(batch_num):
			batch_train = X[i*batch_size : min((i + 1) * batch_size, N)]
			batch_target = Y[i*batch_size : min((i + 1) * batch_size, N)]
			feed_dict = {
				self.new: batch_train, 
				self.new_targets: batch_target,
			}
			total += self.sess.run(self.new_loss, feed_dict)
		
		return total / batch_num
	
	
	def predict(self, X, batch_size):
		feed_dict={self.new: X}
		return self.sess.run(self.new_decoded, feed_dict)
	
	
	def save_img(self, epoch, train, decoded, save_num = 10):
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

	
	def save_model(self):
		self.save.save(self.sess, 'model\\my_model')
	
		
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

model = AUTOENCODER(x_shape=[28, 28, 1], y_shape=[28, 28, 1], learning_rate= 0.05,)

model.fit(X=x_train,
   		  Y=x_train,
		  test=x_test,
		  test_target=x_test,
		  epochs=5,
		  batch_size = 16,
		  suffle=1,
		  eval=1,
		  )
		   
model.save_model()