from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt
import hyper_parameter as p

autoencoder = load_model('model/autoencoder.h5')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

decoded_imgs = autoencoder.predict(x_test)
for i in range(25):
	fig = plt.figure()
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	fig.savefig('predicted-data/'+str(i)+'-encoded.png')
	
	fig = plt.figure()
	plt.imshow(x_test[i].reshape(28, 28))
	fig.savefig('predicted-data/'+str(i)+'-origin.png')