'''
fitting parameters
'''
epochs = 50
batch_size = 128

'''
autoencoder
'''
conv2D_num = 3
filter = [64, 32, 16, 8]
kernal_size = [3, 3, 3, 3]
activation_en = ['relu', 'relu', 'relu', 'relu']
activation_de = ['relu', 'relu', 'relu', 'relu']
padding = ['same', 'same', 'same', 'same']
de_padding = ['same', 'same', 'same', 'same']
max_pooling = [0, 1, 1, 1]
pool_size = [2, 2, 2]