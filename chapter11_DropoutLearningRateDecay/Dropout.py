#Dropout仅在模型训练中使用，在评估中不使用

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
from keras.layers import Embedding

# 导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target

#设置随机种子
seed=7
np.random.seed(seed)

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
    sgd=SGD(lr=0.01,momentum=0.8,decay=0.0,nesterov=False)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model=KerasClassifier(build_fn=create_model,epochs=200,batch_size=10,verbose=2)


kfold=KFold(n_splits=5,shuffle=True,random_state=seed)
results=cross_val_score(model,x,Y,cv=kfold)
print('Accuracy : %.2f%% (%.2f)'%(results.mean()*100,results.std()))