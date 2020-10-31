from pyimagesearch import datasets
from pyimagesearch import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import tensorflow as tf
from tensorflow import feature_column

import numpy as np
import argparse
import locale
import os
import cv2
#%%
#grab ROI for as input of predict model
import glob
import os
from os import walk
names = ["770L01.png","770L02.png","770R01.png","770R02.png","850L01.png","850L02.png","850R01.png","850R02.png","940L01.png","940L02.png","940R01.png","940R02.png"]
col = 2
r = 2

for sub in os.listdir(r"demo\data"):
    path = r"demo\data"
    save_path = r"demo\wrist"# a path for saving image

    path = os.path.join(path, sub)
    save_path = os.path.join(save_path, sub)
    if not os.path.isfile(save_path):
        os.makedirs(save_path)
    # print(path)

    cv_img = []
    i = 0
    a = 0
    for img in os.listdir(path):
        if os.path.join(path, img).endswith(".png"):
            img = cv2.imread(os.path.join(path, img))
            cv_img.append(img)
                                      
        
            #Do histogram equalization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #turn RGB into GRAY 
            hist,bins = np.histogram(gray.flatten(),256,[0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max()/ cdf.max()
            cdf_m = np.ma.masked_equal(cdf,0)# 除去直方圖中的0值
            cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
            cdf = np.ma.filled(cdf_m,0).astype('uint8')# 將掩模處理掉的元素補為0
            img2 = cdf[gray.astype(np.uint8)]
        
            # blur_gray = cv2.GaussianBlur(img2, (101, 101), 0) # Gaussian filter, the kernel must be an odd number
            ret,thresh1 = cv2.threshold(img2,200,255,cv2.THRESH_BINARY)
        
            _, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
            try: hierarchy = hierarchy[0]
            except: hierarchy = []
        
            height, width = thresh1.shape
            min_x, min_y = width, height
            max_x = max_y = 0
        
            # computes the bounding box for the contour, and draws it on the frame,
            for contour, hier in zip(contours, hierarchy):
                (x,y,w,h) = cv2.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x+w, max_x)
                min_y, max_y = min(y, min_y), max(y+h, max_y)
        
        
            if max_x - min_x > 0 and max_y - min_y > 0:
                cv2.rectangle(img, (int(min_x*1.1), int(min_y*1.0)), (int(max_x*0.95), int(max_y*0.9)), (255, 0, 0), 2) # 畫出適當ROI
            
            x_range = int(max_x*0.95) - int(min_x*1.1)
            if int(max_y*0.9) - (int(min_y) + x_range) < abs(int(min_x*1.1) - int(max_x*0.95))/5:
                add = int(max_y*0.9) - int(min_y) - abs(int(min_x*1.1) - int(max_x*0.95))/3
                rect =img2 [(int(min_y) + int(add)):int(max_y*0.9), int(min_x*1.1):int(max_x*0.95)]  
                
            else:
                rect =img2 [(int(min_y) + x_range):int(max_y*0.9), int(min_x*1.1):int(max_x*0.95)]  
            
            cv2.imwrite(os.path.join(save_path, "{}".format(names[a])),rect)
            a += 1
            if a == 12:
                a = 0
        
            if col <= 7 :        
                ws.cell (row = r, column = col).value = rect.mean()
                col += 1
            else :
                col = 2
                r += 1        
                ws.cell (row = r, column = col).value = rect.mean()
                col += 1
#%%

df = datasets.load_data(r'C:\Users\User\Desktop\Peter\Bone_density\demo\demo.xlsx')
df.dropna(subset = ["Name"], inplace=True)
df = df.reset_index(drop=True)
df.pop("Name")

images_Left = datasets.load_wrist_images(df,r'C:\Users\User\Desktop\Peter\Bone_density\demo\wrist',left_right = "Left")

#%%
feature_columns = []
feature_layer_inputs = {}
  
sex = feature_column.categorical_column_with_vocabulary_list(
      'Sex', ['male', 'female'])
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)
feature_layer_inputs['Sex'] = tf.keras.Input(shape=(1,), name='Sex', dtype=tf.string)

  
age = feature_column.numeric_column("Age")
age_buckets = feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70])
feature_columns.append(age_buckets)
# demo(age_buckets)

Menopause = feature_column.categorical_column_with_vocabulary_list(
      'Menopause', ['not suit', 'yes', 'no'])
Menopause_embedding = feature_column.embedding_column(Menopause, dimension=6)
feature_columns.append(Menopause_embedding)
feature_layer_inputs['Menopause'] = tf.keras.Input(shape=(1,), name='Menopause', dtype=tf.string)
# demo(Menopause_embedding)


Bone_injured = feature_column.categorical_column_with_vocabulary_list(
      'Bone_injured', ['yes', 'no'])
Bone_injured_one_hot = feature_column.indicator_column(Bone_injured)
feature_columns.append(Bone_injured_one_hot)
# Bone_injured_embedding = feature_column.embedding_column(Bone_injured, dimension=8)
feature_layer_inputs['Bone_injured'] = tf.keras.Input(shape=(1,), name='Bone_injured', dtype=tf.string)
# demo(Bone_injured_one_hot)

#%%
test_data = []
first = True
for feature in feature_columns:
    feature_layer = tf.keras.layers.DenseFeatures(feature)
    feature_array = feature_layer(dict(df)).numpy()
    if first:
        test_data=feature_array
        first = False
        continue
    test_data = np.concatenate((test_data, feature_array), axis=1)
    print(feature_layer(dict(df)).numpy())
    
#%%
import keras
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

mlp = models.create_mlp(np.asarray(test_data).shape[1], regress=True)
cnn_left = models.create_cnn(256, 128, 6, regress=False)
# cnn_right = models.create_cnn(256, 128, 6, regress=False)



# create the input to our final set of layers as the *output* of both
# the MLP and CNN
# combinedInput = concatenate([mlp.output, cnn_left.output, cnn_right.output])
combinedInput = concatenate([mlp.output, cnn_left.output])
 
# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(8, activation=LeakyReLU(alpha=0.2))(combinedInput)
x = Dense(4, activation=LeakyReLU(alpha=0.2))(x)
x = Dense(1)(x)
 
# our final model will accept categorical/numerical data on the MLP
# model = Model(inputs=[mlp.input, cnn_left.input, cnn_right.input], outputs=x)
my_model = Model(inputs=[mlp.input, cnn_left.input], outputs=x)

my_model.load_weights('Radius_UD_L.h5')
my_model.summary()


#%%
predictions = my_model.predict([test_data, images_Left])

print(predictions)













