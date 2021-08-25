import pandas as pd
import random
import numpy as np


dfval = pd.read_csv('./data/testing_new.csv')
dfval = dfval[dfval.groupby('stay_id')['stay_id'].transform('size') >= 5]


min_len = 10
max_len = 61
max_test_len = 20

dftrain = pd.read_csv('./data/training_new.csv')

def get_train_data_label(df):
    x = []
    y = []
    stay_ids = list(set(df['stay_id'].values.tolist()))
    for id in stay_ids:
        sub_data = df[df['stay_id'].isin([id])].values.tolist()
        if len(sub_data) < max_len:
            while len(sub_data)<max_len:
                sub_data.append(random.choice(sub_data))
        if len(sub_data) > max_len:
            sub_data = random.sample(sub_data,max_len)
        x.append(sub_data)
    random.shuffle(x)
    for i in range(len(x)):
        y.append(x[i][0][-1])
        for j in range(len(x[0])):
            x[i][j] = x[i][j][1:-1]
    a = 1
    return np.array(x),np.array(y)


def get_data_label(df):
    x = []
    y = []
    stay_ids = list(set(df['stay_id'].values.tolist()))
    for id in stay_ids:
        sub_data = df[df['stay_id'].isin([id])].values.tolist()[:max_test_len]
        if len(sub_data) < max_len:
            while len(sub_data)<max_len:
                sub_data.append(random.choice(sub_data))
        if len(sub_data) > max_len:
            sub_data = random.sample(sub_data,max_len)
        x.append(sub_data)
    random.shuffle(x)
    for i in range(len(x)):
        y.append(x[i][0][-1])
        for j in range(len(x[0])):
            x[i][j] = x[i][j][1:-1]
    a = 1
    return np.array(x),np.array(y)

x_t,y_t = get_train_data_label(dftrain)
x_v,y_v = get_data_label(dfval)

np.save('./data/x_t.npy',x_t)
np.save('./data/y_t.npy',y_t)
np.save('./data/x_v.npy',x_v)
np.save('./data/y_v.npy',y_v)