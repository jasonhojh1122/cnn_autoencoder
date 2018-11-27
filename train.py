from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt
import hyper_parameter as p
import time


def generate_autoencoder(start):
	x = Conv2D(p.filter[0], p.kernal_size[0], 
			   activation=p.activation_en[0], padding=p.padding[0], name='conv2D-0')(start)
	if p.max_pooling[0]:
		x = MaxPooling2D(p.pool_size[0], padding=p.padding[0], name='maxPooling-0')(x)
	
	for l in range(1, p.conv2D_num):
		x = Conv2D(p.filter[l],
				   p.kernal_size[l],
				   activation=p.activation_en[l],
				   padding=p.padding[l],
				   name='conv2D-'+str(l))(x)
		if p.max_pooling[l]:
			x = MaxPooling2D(p.pool_size[l], padding=p.padding[l], name='maxPooling-'+str(l))(x)
	
	for l in range(p.conv2D_num-1, -1, -1):
		x = Conv2D(p.filter[l],
				   p.kernal_size[l],
				   activation=p.activation_de[l],
				   padding=p.de_padding[l],
				   name='deconv2D-'+str(l))(x)
		if p.max_pooling[l]:
			x = UpSampling2D(p.pool_size[l])(x)
	
	decoded = Conv2D(1, p.kernal_size[0], activation='sigmoid', padding='same', name='decoded')(x)
	
	return decoded


class save_data(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		# epoch integer, logs = {val_loss: , loss: }
		encoded_layer_model = Model(inputs=autoencoder.input,
                                    outputs=autoencoder.get_layer('deconv2D-'+str(p.conv2D_num-1)).output)
		decoded_layer_model = Model(inputs=autoencoder.input,
									outputs=autoencoder.output)
		encoded_imgs = encoded_layer_model.predict(x_test)
		decoded_imgs = decoded_layer_model.predict(x_test)
		n = 10
		fig = plt.figure(figsize=(20, 8))
		for i in range(n):
			ax = plt.subplot(1, n, i+1)
			plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		fig.savefig('encoded-'+str(epoch)+'.png')


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

saveDataCallback = save_data()

input_img = Input(shape=(28, 28, 1))
decoded = generate_autoencoder(input_img)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

history = autoencoder.fit(x_train, x_train,
                epochs=p.epochs,
                batch_size=p.batch_size,
                shuffle=True,
                validation_data=(x_test, x_test))


with open('training-data/loss.txt', 'w') as f:
	all_loss = history.history['loss']
	for loss in all_loss:
		f.write(str(loss)+'\n')

with open('training-data/val_loss.txt', 'w') as f:
	all_loss = history.history['val_loss']
	for loss in all_loss:
		f.write(str(loss)+'\n')
		
autoencoder.save('model/autoencoder.h5')
