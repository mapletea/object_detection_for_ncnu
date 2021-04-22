from data import load_data, load_img_data
import numpy as np
import cv2
# from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    # plt.ylabel('Accuracy')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],
          'predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(128,128,3)))
    plt.show()
    for j in range(8):
        print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

# input
start = time.time()
(x_train,y_train),(x_test,y_test)=load_img_data()

x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)

# # replace G([1]) with 0
# for i in range(len(x_train)):
#     x_train[i,:,:,1] = x_train[i,:,:,1].astype('float32') / 5
#     # x_train[i,:,:,1] = 0

# for i in range(len(x_test)):
#     x_test[i,:,:,1] = x_test[i,:,:,1].astype('float32') / 5
#     # x_test[i,:,:,1] = 0

print(time.time() - start)
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

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
# plot_model(model, to_file='..\\data\\Sequential_Model.png', show_shapes=True)
# from PIL import Image
# image = Image.open('..\\data\\Sequential_Model_noG.png')
# image.show()
print(model.summary())

start = time.time()
# train
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=10, batch_size=128, verbose=1)
print("Execution Time: ", time.time() - start)

# show training history
show_train_history('accuracy','val_accuracy')
show_train_history('loss','val_loss')

# Step 6. 評估模型準確率
scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=0)
print(scores[:10])

# 進行預測

prediction=model.predict(x_test_normalize)
prediction=np.argmax(prediction,axis=1)
prediction[:10]

print("prediction")
print(prediction.shape)
print(prediction)

# 查看預測結果

# label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
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
plot_images_labels_prediction(x_test,y_test,prediction,0,10)

# 查看預測機率

Predicted_Probability=model.predict(x_test_normalize)

show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,0)
show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,3)

# Step 8. Save Weight to h5 

model.save_weights("./buildingModel_normal.h5")
print("Saved model to disk")