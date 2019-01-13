from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

#设置随机数种子
seed=89
np.random.seed(seed)

#导入数据
dataset=datasets.load_iris()

X=dataset.data
y=dataset.target

x_train,x_incre,y_train,y_incre=train_test_split(X,y,test_size=0.2,random_state=seed)

#将标签转换
Y_train_labels=to_categorical(y_train,num_classes=3)

#构建模型函数
def create_model(optimizer='adam',init='glorot_uniform'):
    model=Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

#构建模型
model=create_model()
model.fit(x_train,Y_train_labels,epochs=10,batch_size=5,verbose=2)

scores=model.evaluate(x_train,Y_train_labels,verbose=0)
print('Increment %s:%.2f%%'%(model.metrics_names[1],scores[1]*100))

#将模型保存成JSON文件
model_json=model.to_json()
with open('model.incre.json','w') as file:
    file.write(model_json)

#保存模型的权重
model.save_weights('model.incre.json.h5')

#从JSON读取文件
with open('model.incre.json','r') as file:
    model_json_read=file.read()

#加载模型
new_model=model_from_json(model_json_read)
new_model.load_weights('model.incre.json.h5')

#编译模型
new_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#增量训练模型
Y_incre_labels=to_categorical(y_incre,num_classes=3)
new_model.fit(x_incre,Y_incre_labels,epochs=10,batch_size=5)

scores=new_model.evaluate(x_incre,Y_incre_labels)
print('Increment %s:%.2f%%'%(new_model.metrics_names[1],scores[1]*100))




