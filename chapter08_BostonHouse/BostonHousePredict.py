from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#导入数据
dataset=datasets.load_boston()

X=dataset.data
y=dataset.target


#设置随机数字
seed=7
np.random.seed(seed)

#构建模型函数
def create_model(units_list=[13],optimizer='adam',init='normal'):
    model=Sequential()

    #构建第一个隐藏层和输入层
    unints=units_list[0]
    model.add(Dense(units=unints,activation='relu',input_dim=13,kernel_initializer=init))
    #构建更多的隐藏层
    for unints in units_list[1:]:
        model.add(Dense(units=unints,activation='relu',kernel_initializer=init))

    #最后一层,输出预测值
    model.add(Dense(units=1,kernel_initializer=init))
    #编译模型
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    return model


model=KerasRegressor(build_fn=create_model,epochs=200,batch_size=5,verbose=0)

#设置算法评估基准
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)
print('Baseline :%.2f (%.2f) MSE:'%(results.mean(),results.std()))


#对输入数据集进行标准化处理
steps=[]
steps.append(('standardize',StandardScaler()))
steps.append(('mlp',model))
pipeline=Pipeline(steps)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(model,X,y,cv=kfold)
print('Standardize:%.2f (%.2f) MSE'%(results.mean(),results.std()))


#GridSearch进行参数搜索
param_grid={}
param_grid['units_list']=[[20],[13,6]]
param_grid['optimizer']=['adam','rmsprop']
param_grid['init']=['glorot_uniform','normal']
param_grid['epochs']=[100,200]
param_grid['batch_size']=[5,20]

#调参
scaler=StandardScaler()
scaler_x=scaler.fit_transform(X)
grid=GridSearchCV(estimator=model,param_grid=param_grid)
results=grid.fit(scaler_x,y)

#输出结果
print('Best:%f using %s'%(results.best_score_,results.best_params_))
means=results.cv_results_['mean_test_score']
stds=results.cv_results_['std_test_score']
params=results.cv_results_['params']

for mean,std,param in zip(means,stds,params):
    print('%f (%f) with:%r'%(mean,std,param))
