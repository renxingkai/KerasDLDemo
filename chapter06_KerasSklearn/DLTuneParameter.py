#使用GridSearch进行参数的搜索
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

#构建模型
def create_model(optimizer='adam',init='glorot_uniform'):
    model=Sequential()
    model.add(Dense(units=12,kernel_initializer=init,input_dim=8,activation='relu'))
    model.add(Dense(units=8,kernel_initializer=init,activation='relu'))
    model.add(Dense(units=1,kernel_initializer=init,activation='sigmoid'))

    #模型编译
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

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
model=KerasClassifier(build_fn=create_model,verbose=0)

#构建需要调参的参数
param_grid={}
param_grid['optimizer']=['rmsprop','adam']
param_grid['init']=['glorot_uniform','normal','uniform']
param_grid['epochs']=[50,100,150,200]
param_grid['batch_size']=[5,10,20]

#调参
grid=GridSearchCV(estimator=model,param_grid=param_grid)
results=grid.fit(X,y)

print('Best:%f using %s'%(results.best_score_,results.best_params_))
means=results.cv_results_['mean_test_score']
stds=results.cv_results_['std_test_score']
params=results.cv_results_['params']

for mean,std,param in zip(means,stds,params):
    print('%f (%f) with:%r'(mean,std,param))
























