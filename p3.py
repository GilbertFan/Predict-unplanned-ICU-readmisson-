import pandas as pd
import random
import numpy as np


x_t = np.load('./data/x_t.npy')
y_t = np.load('./data/y_t.npy')
x_v = np.load('./data/x_v.npy')
y_v = np.load('./data/y_v.npy')

batch_size = 20

def get_x_y(df,df1):
    res_x = []
    res_y = []
    df = df.tolist()
    if len(df) != len(df1):
        print('error')
    while len(df)>batch_size:
        res_x_sub = df[:batch_size]
        df = df[batch_size:]
        res_x.append(res_x_sub)


        res_y_sub = df1[:batch_size]
        df1 = df1[batch_size:]
        res_y.append(res_y_sub)

    return res_x,res_y

x_t1,y_t1 = get_x_y(x_t,y_t)
x_v1,y_v1 = get_x_y(x_v,y_v)

np.save('./data/x_t1.npy',x_t1)
np.save('./data/y_t1.npy',y_t1)
np.save('./data/x_v1.npy',x_v1)
np.save('./data/y_v1.npy',y_v1)