from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier

#构建模型
def create_model():
    model=Sequential()
    model.add(Dense(units=12,input_dim=8,activation='relu'))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))

    #模型编译
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

#随机种子
seed=56

np.random.seed(seed)

file="pima-indians-diabetes.csv"

#导入数据
dataset=np.loadtxt(file,delimiter=',')

#分割数据
X=dataset[:,0:8]
y=dataset[:,8]

#创建for sklearn的模型
model=KerasClassifier(build_fn=create_model,epochs=150,batch_size=10,verbose=0)

#10折交叉验证
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)

print(results.mean())






















