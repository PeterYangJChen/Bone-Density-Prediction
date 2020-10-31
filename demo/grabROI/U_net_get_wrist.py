from model import *
from data import *
import matplotlib.pyplot as  plt
import numpy as np
import glob
import os
from PIL import Image

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
# Input test data
file_path = r'C:\Users\User\Desktop\Peter\Bone_density\demo\data'
name = ["770L01.png","770L02.png","770R01.png","770R02.png","850L01.png","850L02.png","850R01.png","850R02.png","940L01.png","940L02.png","940R01.png","940R02.png"]
for fileName in os.listdir(file_path):
    img_array = []  
    testGene = testGenerator(os.path.join(file_path,fileName),img_array,target_size = (240,320)) # data
     
    # imput unet structure
    model = unet(input_size = (240,320,1)) # model
     
    # Import trained weight
    model.load_weights(r'C:\Users\User\Desktop\Peter\Bone_density\case2\train\grab_Wrist.hdf5')
     
    # output predict result and save
    results = model.predict_generator(testGene,12,verbose=1) # keras
    saveResult(os.path.join(file_path,fileName),results,img_array,name)
    



















