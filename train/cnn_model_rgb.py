from data import load_data, load_img_data
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import cv2
from tensorflow.keras.utils import plot_model
import time

# load data
(x_train,y_train),(x_test,y_test) = load_img_data()

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)

image_size = 128
# image
x_train_b = np.empty((len(x_train),image_size, image_size, 1),dtype="uint8")
x_train_g = np.empty((len(x_train),image_size, image_size, 1),dtype="uint8")
x_train_r = np.empty((len(x_train),image_size, image_size, 1),dtype="uint8")    
x_test_b = np.empty((len(x_test),image_size, image_size, 1),dtype="uint8")
x_test_g = np.empty((len(x_test),image_size, image_size, 1),dtype="uint8")
x_test_r = np.empty((len(x_test),image_size, image_size, 1),dtype="uint8")

for i in range(len(x_train)):
    x_train_b[i] = x_train[i,:,:,0].reshape((image_size,image_size,1))
    x_train_g[i] = x_train[i,:,:,1].reshape((image_size,image_size,1))
    x_train_r[i] = x_train[i,:,:,2].reshape((image_size,image_size,1))

for i in range(len(x_test)):
    x_test_b[i] = x_test[i,:,:,0].reshape((image_size,image_size,1))
    x_test_g[i] = x_test[i,:,:,1].reshape((image_size,image_size,1))
    x_test_r[i] = x_test[i,:,:,2].reshape((image_size,image_size,1))

x_train_normalize_b = x_train_b.astype('float32') / 255.0
x_test_normalize_b = x_test_b.astype('float32') / 255.0

x_train_normalize_g = x_train_g.astype('float32') / 255.0
x_test_normalize_g = x_test_g.astype('float32') / 255.0

x_train_normalize_r = x_train_r.astype('float32') / 255.0
x_test_normalize_r = x_test_r.astype('float32') / 255.0

x_train_normalize = {
    'img_b_input': x_train_normalize_b,
    'img_g_input': x_train_normalize_g,
    'img_r_input': x_train_normalize_r,
}

x_test_normalize = {
    'img_b_input': x_test_normalize_b,
    'img_g_input': x_test_normalize_g,
    'img_r_input': x_test_normalize_r,
}

y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

print(y_train_OneHot.shape)
print(y_train_OneHot)
print(y_test_OneHot.shape)

# layer setting

img_b_input = keras.Input(shape=(128, 128, 1), name='img_b_input')
hidden_b = layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, activation='relu', name='hidden_b', padding='same')(img_b_input)
pool_b = layers.MaxPooling2D(pool_size=(2, 2), name='pool_b')(hidden_b)
hidden_b_ft = layers.Flatten()(pool_b)

img_g_input = keras.Input(shape=(128, 128, 1), name='img_g_input')
hidden_g = layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, activation='relu', name='hidden_g', padding='same')(img_g_input)
pool_g = layers.MaxPooling2D(pool_size=(2, 2), name='pool_g')(hidden_g)
hidden_g_ft = layers.Flatten()(pool_g)

img_r_input = keras.Input(shape=(128, 128, 1), name='img_r_input')
hidden_r = layers.Conv2D(filters=8, kernel_size=(3,3), strides=1, activation='relu', name='hidden_r', padding='same')(img_r_input)
pool_r = layers.MaxPooling2D(pool_size=(2, 2), name='pool_r')(hidden_r)
hidden_r_ft = layers.Flatten()(pool_r)

concat = layers.Concatenate()([hidden_b_ft, hidden_g_ft, hidden_r_ft])

hidden2 = layers.Dense(512, activation='relu', name='hidden2')(concat)
dropout2 = layers.Dropout(rate=0.25)(hidden2)

hidden3 = layers.Dense(512, activation='relu', name='hidden3')(dropout2)
dropout3 = layers.Dropout(rate=0.25)(hidden3)

hidden4 = layers.Dense(512, activation='relu', name='hidden4')(dropout3)
dropout4 = layers.Dropout(rate=0.25)(hidden4)

outputs = layers.Dense(8, activation='softmax', name='Output')(dropout4)
model = keras.Model(inputs=[img_b_input, img_g_input, img_r_input], outputs=outputs)
# plot_model(model, to_file='..\\data\\Sequential_Model.png', show_shapes=True)
# from PIL import Image
# image = Image.open('..\\data\\Sequential_Model.png')
# image.show()
print(model.summary())

# train
start = time.time()
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=10, batch_size=64, verbose=1)
print("Exection time: ", time.time() - start)
# test
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    # plt.ylabel('Accuracy')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# show_train_history('acc','val_acc')
show_train_history('accuracy','val_accuracy')
show_train_history('loss','val_loss')


# Step 6. 評估模型準確率

scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=0)
print(scores[:10])


# 進行預測

# prediction=model.predict(x_test_normalize)
# prediction=np.argmax(prediction,axis=1)
# prediction[:10]

# print("prediction")
# print(prediction.shape)
# print(prediction)

# # 查看預測結果

# # label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
# label_dict={
#   0:'science', # 科院
#   1:'manage', # 管院
#   2:'edu', # 教院
#   3:'human', # 人院
#   4:'admin', # 行政大樓
#   5:'library', # 圖書館
#   6:'activity', # 學活
#   7:'restaurant', # 學餐
# }
# print(label_dict)		

# import matplotlib.pyplot as plt
# def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
#     fig = plt.gcf()
#     fig.set_size_inches(12,14)
#     if num>25: num=25 
#     for i in range(0, num):
#         ax=plt.subplot(5,5, 1+i)
#         ax.imshow(images[idx],cmap='binary')
                
#         title=str(i)+','+label_dict[labels[i]]
#         if len(prediction)>0:
#             title+='=>'+label_dict[prediction[i]]
            
#         ax.set_title(title,fontsize=10)
#         ax.set_xticks([]);ax.set_yticks([])        
#         idx+=1 
#     plt.show()

# plot_images_labels_prediction(x_test,y_test,prediction,0,10)

# # 查看預測機率

# Predicted_Probability=model.predict(x_test_normalize)

# def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
#     print('label:',label_dict[y[i]],
#           'predict:',label_dict[prediction[i]])
#     plt.figure(figsize=(2,2))
#     plt.imshow(np.reshape(x_test[i],(128,128,3)))
#     plt.show()
#     for j in range(8):
#         print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

# show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,0)
# show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,3)

# Step 8. Save Weight to h5 

model.save_weights("./buildingModel_rgb.h5")
print("Saved model to disk")