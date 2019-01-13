from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras import backend
from keras.layers import Dropout
backend.set_image_data_format('channels_first')

#导入数据
(X_train,y_train),(X_validation,y_validation)=mnist.load_data()

#显示四张手写数据集
plt.subplot(221)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))

# plt.show()
#设定随机种子
seed=777
np.random.seed(seed)

#卷积NN感知机模型
#(60000, 28, 28)
print(X_train.shape)

num_pixels=X_train.shape[1]*X_train.shape[2]

print(num_pixels)

X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_validation=X_validation.reshape(X_validation.shape[0],1,28,28).astype('float32')

#格式化数据到0-1
X_train=X_train/255
X_validation=X_validation/255

#进行One-hot编码
y_train=np_utils.to_categorical(y_train)
y_validation=np_utils.to_categorical(y_validation)

num_classes=y_validation.shape[1]
print(num_classes)

#定义基础MLP模型
def create_model():
    #创建模型
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=10,activation='softmax'))
    #编译模型
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model=create_model()
model.fit(X_train,y_train,epochs=10,batch_size=200,verbose=2)
score=model.evaluate(X_validation,y_validation)
print('CNN Small:%.2f%%'%(score[1]*100))
#MLP:98.37%