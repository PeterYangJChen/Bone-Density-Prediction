# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import glob
import cv2
import os
#%%


def load_data(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	df = pd.read_excel(inputPath)
	return df

def df_to_dataset(dataframe, batch_size, shuffle=True,target = 'Wrist BMD'):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def hist_eq(gray):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turn RGB into GRAY 
    hist,bins = np.histogram(gray.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)# 除去直方圖中的0值
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')# 將掩模處理掉的元素補為0
    img2 = cdf[gray.astype(np.uint8)]
    return img2

def load_wrist_images(df, inputPath,left_right = 'Left'):
	# initialize our images array (i.e., the house images themselves)
    images = []
    Wrist_image = []
    #df.index.valuesrepresent the index of sub
    for img_file in os.listdir(inputPath):
        wristPath = os.path.join(inputPath, img_file)
        output = []
        for img in os.listdir(wristPath):
            if img.endswith('png'):
                if img[3] == left_right[0]:#check if input image match the hand we required
                    image = cv2.imread(os.path.join(wristPath, img), cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (256, 128))
                    image = np.asarray(hist_eq(image))
                    output.append(image)
        Wrist_image.append(output)
            
    # for i in df.index.values:
    #     basePath = os.path.sep.join([inputPath, "sub_{}\*".format(i + 1)])
    #     wristPath = list(glob.glob(basePath))
        
    #     output = []
    #     for wrist in wristPath:
    #         if left_right:#check whether we wanna distinct left_right hand
    #             if  wrist[-7] == left_right[0]:#check if input image match the hand we required
    #                 image = cv2.imread(wrist)
    #                 image = cv2.resize(image, (256, 128))
    #                 image = hist_eq(image)
    #                 output.append(image)
    #         else:
    #             image = cv2.imread(wrist)
    #             image = cv2.resize(image, (256, 128))
    #             image = hist_eq(image)
    #             output.append(image)
    #     Wrist_image.append(output)

    # return our set of images
    out_img = np.asarray(Wrist_image).swapaxes(1,3)
    out_img = out_img.swapaxes(1,2)
    return out_img
    # return np.asarray(tf.reshape(Wrist_image,[max(df.index.values)+1,256,128,-1]))




















