from data import *
from model import *
import matplotlib.pyplot as  plt
import numpy as np
import glob
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf 

#%%
#allocate GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
#%%
#Generate train data with size (2,W,H,1) and val data with size (1,W,H,1)
gen_path = 'result/'
#data augmentation
data_gen_args = dict(rotation_range=0.2, 
                    width_shift_range=0.05, 
                    height_shift_range=0.05, 
                    shear_range=0.05, 
                    zoom_range=0.05, 
                    horizontal_flip=True,  
                    fill_mode='nearest') 
target_size = (240,320)

trainGene = trainGenerator(2,'train/','wrist','train/',data_gen_args,target_size=target_size)
valGene = trainGenerator(2,'val/','wrist','val/',data_gen_args,target_size=target_size)

#%%
from keras.callbacks import TensorBoard
#train data with unet
model = unet()
#train model with callback
history=model.fit_generator(trainGene,
                    steps_per_epoch=16,
                    epochs=1000,
                    validation_data=valGene,
                    validation_steps=4,
                    callbacks=[
                    EarlyStopping(monitor='val_dice_coef', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_dice_coef', factor=0.1, patience=5)
                ])

#%%
#plot accuracy
import matplotlib.pyplot as plt

acc=history.history['dice_coef']
val_acc=history.history['val_dice_coef']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'b',color='green',label='Training dice_coef')
plt.plot(epochs,val_acc,'b',label='Validation val_dice_coef')
plt.title('Training and Validation dice_coef')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'go--',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
#%%
model.save('grab_Wrist.hdf5')

