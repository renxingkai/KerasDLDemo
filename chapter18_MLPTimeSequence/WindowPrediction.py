import pandas as pd
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
import math
import numpy as np


# MLP
SEED = 777
BATCH_SIZE = 2
EPOCHS = 200
filename = 'international-airline-passengers.csv'
footer = 3
look_back = 3


def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # 将前一天作为训练集，后一天作为预测集
        x = dataset[i:i + look_back, 0]
        dataX.append(x)
        y = dataset[i + look_back, 0]
        dataY.append(y)
        print('X:%s,Y:%s' % (x, y))
    return np.array(dataX), np.array(dataY)


def build_model():
    model = Sequential()
    model.add(Dense(units=8, input_dim=look_back, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    #设置随机数
    np.random.seed(SEED)
    #导入数据
    # 导入数据
    data = pd.read_csv(filename, usecols=[1], engine='python', skipfooter=footer)
    dataset=data.values.astype('float32')
    print(dataset)
    train_size=int(len(dataset)*0.67)
    validation_size=len(dataset)-train_size
    train,validation=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
    #创建dataset，让数据出现相关性
    X_train,y_train=create_dataset(train)
    X_validation,y_validation=create_dataset(validation)
    print(X_validation)
    print('--'*30)
    print(y_validation)
    #训练模型
    model=build_model()
    model.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2)
    #评估模型
    train_score=model.evaluate(X_train,y_train,verbose=0)
    print('Train score:%.2f MSE(%.2f RMSE)'%(train_score,math.sqrt(train_score)))
    #验证集评估模型
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Validation score:%.2f MSE(%.2f RMSE)' % (validation_score, math.sqrt(validation_score)))
    #利用图表查看预测趋势
    predict_train=model.predict(X_train,batch_size=16)
    predict_validation=model.predict(X_validation,batch_size=16)
    #构建通过训练数据集进行预测的图表数据
    predict_train_plot=np.empty_like(dataset)
    predict_train_plot[:,:]=np.nan
    predict_train_plot[look_back:len(predict_train)+look_back,:]=predict_train
    #构建通过评估数据集进行预测的图表数据
    predict_validation_plot= np.empty_like(dataset)
    predict_validation_plot[:,:] = np.nan
    predict_validation_plot[len(predict_train)+look_back*2+1:len(dataset)-1,:] = predict_validation

    #图表显示
    plt.plot(dataset,color='blue')
    plt.plot(predict_train_plot,color='green')
    plt.plot(predict_validation_plot,color='red')
    plt.show()





