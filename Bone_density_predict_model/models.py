from keras.models import Sequential
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras import layers
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#%% 
def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(32, input_dim=dim, activation=LeakyReLU(alpha=0.2)))
	model.add(Dense(16, activation=LeakyReLU(alpha=0.2)))
	model.add(Dense(8, activation=LeakyReLU(alpha=0.2)))
 
	# return our model
	return model

def create_cnn(width, height, depth,regress=False):
 	# initialize the input shape and channel dimension, assuming
 	# TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    
    # define the model input
    inputs = Input(shape=inputShape)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    x1 = BatchNormalization(axis=chanDim)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    x2 = BatchNormalization(axis=chanDim)(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    x3 = BatchNormalization(axis=chanDim)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation ='relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    x = Flatten()(pool4)
    x = Dense(16,activation="relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    
   	# apply another FC layer, this one to match the number of nodes
   	# coming out of the MLP
    x = Dense(8,activation="relu")(x)
    
   	# construct the CNN
    model = Model(inputs, x)
    
   	# return the CNN
    return model

