#学习率衰减分为线性衰减和指数衰减
#指数学习率衰减
#一般通过在固定的epoch周期将学习率降低50%
from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from math import pow,floor
from keras.callbacks import LearningRateScheduler

# 导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

#设置随机种子
seed=7
np.random.seed(seed)

#计算学习率
def step_decay(epoch):
    init_lrate=0.1
    drop=0.5
    #每10个epoch衰减0.5
    epochs_drop=10
    lrate=init_lrate*pow(drop,floor(1+epoch)/epochs_drop)
    return lrate

#线性衰减学习率
#构建模型函数
def create_model(init='glorot_uniform'):
    model=Sequential()
    #鸢尾花数据集的4个属性
    model.add(Dropout(rate=0.2,input_shape=(4,)))
    model.add(Dense(units=4,activation='relu',kernel_initializer=init))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    #在隐藏层使用Dropout
    model.add(Dropout(rate=0.2))
    #最后一层进行分类
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
    #定义优化器
    learning_rate=0.1
    momentum=0.9
    #衰减率
    decay_rate=0
    sgd=SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


#学习率衰减回调
lrate=LearningRateScheduler(step_decay)

model=KerasClassifier(build_fn=create_model,epochs=200,batch_size=10,verbose=2,callbacks=[lrate])

model.fit(x,Y)
