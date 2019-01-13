from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import backend
import os

backend.set_image_data_format('channels_first')


imgGen=ImageDataGenerator()

#从数据集导入数据
(X_train,y_train),(X_validation,y_validation)=mnist.load_data()

#显示9张手写数字
for i in range(0,9):
    #3行 3列 1+i个图
    plt.subplot(331+i)
    plt.imshow(X_train[i],cmap=plt.get_cmap('gray'))

plt.show()


#特征标准化
X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_validation=X_validation.reshape(X_validation.shape[0],1,28,28).astype('float32')

#图像反转
imgGen=ImageDataGenerator(horizontal_flip=True,vertical_flip=True)
imgGen.fit(X_train)
#更改保存位置
file='image'
try:
    os.mkdir(file)
except:
    print('the fold is already exist!')
for X_batch,y_batch in imgGen.flow(X_train,y_train,batch_size=9,save_to_dir=file,save_prefix='oct',save_format='png'):
    for i in range(0,9):
        plt.subplot(331+i)
        plt.imshow(X_batch[i].reshape(28,28),cmap=plt.get_cmap('gray'))
    plt.show()
    break
