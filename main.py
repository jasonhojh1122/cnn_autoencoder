from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import keras
import numpy as np
import matplotlib.pyplot as plt

class toOutput(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		# epoch integer
		# logs = {val_loss: , loss: }
		intermediate_layer_model = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)
		encoded_imgs = intermediate_layer_model.predict(x_train)
		n = 10
		fig = plt.figure(figsize=(20, 8))
		for i in range(n):
			ax = plt.subplot(1, n, i+1)
			# plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
			plt.imshow(encoded_imgs[i].reshape(28,28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		fig.savefig(str(epoch)+'.png')

def generate_encoder(start, layer_name):
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(start)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	return encoded
	
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

layer_name = 'toOutput'
myCallback = toOutput()

input_img = Input(shape=(28, 28, 1))
encoded = generate_encoder(input_img, 'toOutput')
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='toOutput')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[myCallback])
