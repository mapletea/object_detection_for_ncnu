import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
import os
from PIL import Image

label_dict={
  0:'science', # 科院
  1:'manage', # 管院
  2:'edu', # 教院
  3:'human', # 人院
  4:'admin', # 行政大樓
  5:'library', # 圖書館
  6:'activity', # 學活
  7:'restaurant', # 學餐
}

def plot_images_labels_prediction(images, prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')

        if len(prediction)>0:
            title = label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

# load data
source_root = ".\\data\\"
imgs = os.listdir(source_root)
datas = np.empty((len(imgs),3,128, 128),dtype="uint8")

for i in range(len(imgs)):
    img = Image.open(source_root+imgs[i])
    img = img.resize((128,128))
    arr = np.array(img)
    datas[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]

datas = datas.transpose(0, 2, 3, 1)
data_normalize = datas.astype('float32') / 255.0

# model
img_input = keras.Input(shape=(128, 128, 3), name='img_input')
hidden = layers.Conv2D(filters=12, kernel_size=(3,3), strides=1, activation='relu', name='hidden', padding='same')(img_input)
pool = layers.MaxPooling2D(pool_size=(2, 2), name='pool')(hidden)
hidden_ft = layers.Flatten()(pool)

hidden2 = layers.Dense(512, activation='sigmoid', name='hidden2')(hidden_ft)
dropout2 = layers.Dropout(rate=0.25)(hidden2)

hidden3 = layers.Dense(512, activation='relu', name='hidden3')(dropout2)
dropout3 = layers.Dropout(rate=0.25)(hidden3)

hidden4 = layers.Dense(512, activation='relu', name='hidden4')(dropout3)
dropout4 = layers.Dropout(rate=0.25)(hidden4)

outputs = layers.Dense(8, activation='softmax', name='Output')(dropout4)
model = keras.Model(inputs=img_input, outputs=outputs)

# load weight
model.load_weights("./buildingModel_normal.h5")
print("載入模型成功!繼續訓練模型")

# predict
prediction=model.predict(data_normalize)
prediction=np.argmax(prediction,axis=1)
print(prediction)

plot_images_labels_prediction(datas, prediction, 0, num=len(imgs))