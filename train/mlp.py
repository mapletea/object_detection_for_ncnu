#-*- coding: UTF-8 -*- 
import random
#with open('data/array/gray_data_128.csv', 'r') as file:
with open('data/array/data_gray.csv', 'r') as file:
    csv_lines=file.readlines()

trainX=[]
trainy=[]
testX=[]
testy=[]
data = csv_lines[1:]
random.shuffle(data)
for i in range(len(data)):    
    row=data[i].replace('\n', '').split(',')
    if i < len(data)*0.8:
        trainX.append(list(map(int, row[1:])))
        trainy.append(list(map(int, row[0])))
    else:
        testX.append(list(map(int, row[1:])))
        testy.append(list(map(int, row[0])))


dim = len(row[1:])
classes = len(set([j for sub in trainy for j in sub]))

from tensorflow.keras.utils import to_categorical
trainy=to_categorical(trainy, num_classes=classes)
testy=to_categorical(testy, num_classes=classes)

import numpy as np
trainX=np.array(trainX)/255.0
trainy=np.array(trainy)
testX=np.array(testX)/255.0
testy=np.array(testy)
print(trainX.shape)


from keras.models import Sequential
from keras.layers import Dense, Dropout
batchSize = 128
epoch = 35
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(dim,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(classes, activation='softmax'))
print(model.summary())


from tensorflow.keras.optimizers import RMSprop
# 指定 loss function, optimizier, metrics
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'])
              
import datetime
starttime = datetime.datetime.now()
# 指定 batch_size, epochs, validation 後，開始訓練模型
history = model.fit(trainX, trainy,
                    batch_size = batchSize,
                    epochs = epoch,
                    validation_split=0.2,
                    verbose = 1)
endtime = datetime.datetime.now()
print("Execution time:", (endtime - starttime).seconds, "s")

import matplotlib.pyplot as plt
def show_train_history(train_history, which):
    plt.plot(train_history.history[which])
    plt.plot(train_history.history['val_'+which])
    plt.xticks([i for i in range(len(train_history.history[which]))])
    plt.title('Train History')
    plt.ylabel(which)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('pic/dropout4'+which+'.png')
    plt.show()

show_train_history(history, 'acc')
show_train_history(history, 'loss')
print("Train accuracy", history.history['acc'][-1])
print("Validation accuracy", history.history['val_acc'][-1])
print("Train loss", history.history['loss'][-1])
print("Validation loss", history.history['val_loss'][-1])

# 測試集
prediction=model.predict(testX)
testy=np.argmax(testy, axis=1)
prediction=np.argmax(model.predict(testX), axis=1)
print("Actual: ", test_labels)
print("Predict", prediction)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testy, prediction)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testy, prediction, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testy, prediction, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testy, prediction, average='macro')
print('F1 score: %f' % f1)

# 儲存模型
try:
	model.save_weights("mnist.h5")
	print("success")
except:
	print("error")
