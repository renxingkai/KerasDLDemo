from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

# 导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

# Convert labels to categorical one-hot encoding
Y_labels = to_categorical(Y, num_classes=3)

# 设定随机种子
seed = 7
np.random.seed(seed)
# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # 构建模型
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

# 构建模型
model = create_model()

#设置检查点
filepath='weights-improvement-{epoch:02d}-{val_acc:.2f}.h5'
#monitor 为要监视的对象   mode 为对象的模式
checkpoint=ModelCheckpoint(filepath=filepath,monitor='val_acc',mode='max',save_best_only=True)
callback_list=[checkpoint]
model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,callbacks=callback_list)





#仅保存一个文件，模型最优参数值
filepath='weights.best.h5'
#monitor 为要监视的对象   mode 为对象的模式
checkpoint=ModelCheckpoint(filepath=filepath,monitor='val_acc',mode='max',save_best_only=True)
callback_list=[checkpoint]
model.fit(x,Y_labels,validation_split=0.2,epochs=200,batch_size=5,callbacks=callback_list)

#先从检查点恢复模型，然后使用该模型对整个数据集进行预测




