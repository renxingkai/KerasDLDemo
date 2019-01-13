from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.models import  *
from keras.layers import *
from keras.preprocessing import *

#定义常量
SEED=7
TOP_WORDS=5000
MAX_WORDS=500
OUT_DIMENSION=32
BATCH_SIZE=128
EPOCHS=2
RNN_UNITS=100
DROPOUT_RATE=0.2

#构建模型
def build_model():
    model=Sequential()
    model.add(Embedding(input_dim=TOP_WORDS,output_dim=OUT_DIMENSION,input_length=MAX_WORDS))
    #卷积层有助于提取特征
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=RNN_UNITS))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')
    print(model.summary())
    return model


#主函数
if __name__=='__main__':
    np.random.seed(SEED)
    #导入数据
    (X_train,y_train),(X_validation,y_validation)=imdb.load_data(num_words=TOP_WORDS)
    #限定数据集长度
    X_train=sequence.pad_sequences(X_train,maxlen=MAX_WORDS)
    X_validation=sequence.pad_sequences(X_validation,maxlen=MAX_WORDS)
    #生成模型并且训练
    model=build_model()
    model.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2)
    scores=model.evaluate(X_validation,y_validation)
    print('Accuracy :%.2f%%'%(scores[1]*100))