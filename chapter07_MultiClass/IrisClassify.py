from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#导入数据
dataset=datasets.load_iris()

X=dataset.data
y=dataset.target

#设定随机数种子
seed=8
np.random.seed(seed)

#构建模型函数
def create_model(optimizer='adam',init='glorot_uniform'):
    model=Sequential()
    model.add(Dense(4,input_dim=4,activation='relu',kernel_initializer=init))
    model.add(Dense(6,activation='relu',kernel_initializer=init))
    model.add(Dense(3,activation='softmax',kernel_initializer=init))

    #编译模型
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=KerasClassifier(build_fn=create_model,epochs=200,batch_size=5,verbose=0)

#评估模型
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)

print('Accuracy : %.2f%%(%.2f)'%(results.mean()*100,results.std()))

























