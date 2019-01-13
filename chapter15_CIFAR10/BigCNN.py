import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras import backend
from keras.models import Sequential
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.utils import np_utils
backend.set_image_data_format('channels_first')


#设定随机数种子
SEED=9
np.random.seed(SEED)

#导入数据
(X_train,y_train),(X_validation,y_validation)=cifar10.load_data()

#将数据格式化到0-1
X_train=X_train.astype('float32')
X_validation=X_validation.astype('float32')
X_train=X_train/255.
X_validation=X_validation/255.

#进行one-hot编码
y_train=np_utils.to_categorical(y_train)
y_validation=np_utils.to_categorical(y_validation)
num_classes=y_train.shape[1]

#创建模型
def create_model(epochs=25):
    model=Sequential()
    #最大模约束maxnorm
    #input_shape 3通道 32*32
    model.add(Conv2D(32,(3,3),input_shape=(3,32,32),padding='same',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(10,activation='softmax'))
    lrate=0.01
    decay=lrate/epochs
    adam=Adam(lr=lrate,decay=decay)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model

epochs=25
model=create_model(epochs)
model.fit(X_train,y_train,epochs=epochs,batch_size=32,verbose=2)
scores=model.evaluate(x=X_validation,y=y_validation,verbose=0)
print('Accuracy:%.2f%%'%(scores[1]*100))

