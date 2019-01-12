import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from keras.layers import Embedding

#设置随机数种子
np.random.seed(7)

#数据路径
data_file="pima-indians-diabetes.csv"

#导入数据
dataset=np.loadtxt(data_file,encoding='utf-8',delimiter=',')
#分割X和y
X=dataset[:,0:8]
y=dataset[:,8]

#创建模型
model=Sequential()
model.add(Embedding(input_dim=2,output_dim=2))
model.add(Dense(12,activation='relu'))
model.add(Dense(6,activation='relu'))

#最后一层dense，sigmoid输出患病概率
model.add(Dense(1,activation='sigmoid'))

#编译模型
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#训练模型
model.fit(X,y,epochs=10,batch_size=10,validation_split=0.2)

#评估模型
scores=model.evaluate(X,y)
print('\n%s : %.2f%%'%(model.metrics_names[1],scores[1]*100))











