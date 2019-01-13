#使用过去的时间段数据(t-n~t-1)来预测当前的数据t
#将数据按照这种方式进行整理
#这里仅需要预测PM2.5的浓度，因此输出仅保留PM2.5浓度这个项目
#因为风向是类型型数据，需要将风向编码为数字
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def convert_dataset(data, n_input=1, out_index=0, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    # 输入序列 (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 输出结果 (t)
    cols.append(df[df.columns[out_index]])
    names += ['result']
    # 合并输入输出序列
    result = pd.concat(cols, axis=1)
    result.columns = names
    # 删除包含缺失值的行
    if dropnan:
        result.dropna(inplace=True)
    return result


# class_indexs 编码的字段序列号，或者序列号List，列号从0开始
def class_encode(data, class_indexs):
    encoder = LabelEncoder()
    class_indexs = class_indexs if type(class_indexs) is list else [class_indexs]

    values = pd.DataFrame(data).values

    for index in class_indexs:
        values[:, index] = encoder.fit_transform(values[:, index])

    return pd.DataFrame(values) if type(data) is pd.DataFrame else values
























