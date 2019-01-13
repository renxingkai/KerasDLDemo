from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.preprocessing import *
from keras.models import Sequential

# 构建MLP
SEED = 7
MAX_FEATURES = 5000
MAX_WORDS = 500
OUTPUT_DIM = 32
BATCH_SIZE = 128
EPOCHS = 2


def create_model():
    model = Sequential()
    # Embedding
    model.add(Embedding(MAX_FEATURES, OUTPUT_DIM, input_length=MAX_WORDS))
    #一维卷积层
    model.add(Conv1D(filters=32,kernel_size=3,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # 直接展平
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    # 导入数据
    (X_train, y_train), (X_validation, y_validation) = imdb.load_data(num_words=MAX_FEATURES)

    # 合并训练集和验证集
    x = np.concatenate((X_train, X_validation), axis=0)
    y = np.concatenate((y_train, y_validation), axis=0)

    print('shape of x is {}'.format(x.shape))
    print('shape of y is {}'.format(y.shape))

    print('classes is {}'.format(np.unique(y)))

    print('total words:{}'.format(len(np.unique(np.hstack(x)))))

    # 计算单词的平均值和标准差
    result = [len(word) for word in x]
    print('Mean:%.2f words (STD:%.2f)' % (np.mean(result), np.std(result)))

    # 图表展示
    plt.subplot(121)
    plt.boxplot(result)
    plt.subplot(122)
    plt.hist(result)
    # plt.show()
    np.random.seed(SEED)

    #padding
    X_train=sequence.pad_sequences(X_train,maxlen=MAX_WORDS)
    X_validation=sequence.pad_sequences(X_validation,maxlen=MAX_WORDS)
    #生成模型
    model=create_model()
    model.fit(X_train,y_train,validation_data=(X_validation,y_validation),batch_size=BATCH_SIZE,epochs=EPOCHS)
