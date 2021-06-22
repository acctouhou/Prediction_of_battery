# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:06:46 2019

@author: Acc
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm import tqdm


data_dir='dataset'
model_dir='pretrained'

def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    return m,s
def mish(x):
        return x * K.tanh(K.softplus(x))    

def mae(x,y):
    return np.abs(x-y).mean()
def feature_selector(model,x,norm):
    normalized_data=(np.transpose(x,(0,2,1))-norm[0])/norm[1]
    return model.predict(normalized_data,batch_size=128)
def concat_data(x1,x2,x3):
    normalized_data=(np.array(x3)-summary_norm[0])/summary_norm[1]
    return np.hstack((x1,x2,normalized_data))
def re_norm(cell_feature):
    log1=[]
    log2=[]
    for i in range(len(cell_feature)):
        len_=len(cell_feature['%d'%(i)])-100
        for k in range(len_):
            for j in range(0,50,1):            
                log1.append(np.float32(k))
                log2.append(np.float32(eol_data[i]-k))
    log1=np.float32(norm(np.array(log1).reshape(-1,1)))
    log2=np.float32(norm(np.array(log2).reshape(-1,1)))
    return log1,log2
def process2predict(cell_feature):
    x_in=[]
    y_in=[]
    for i in range(len(cell_feature)):
        col1=[]
        col2=[]
        len_=len(cell_feature['%d'%(i)])-100  
        for k in range(len_):
            for j in range(0,50,1):            
                temp=cell_feature['%d'%(i)][k:(j+k+1)]
                col1.append(np.float32(np.pad(temp, ((0,50-j-1),(0,0)), 'edge')))
                col2.append(np.float32(((eol_data[i]-k))-rul_norm[0])/rul_norm[1])
                col2.append((np.float32(k)-s_norm[0])/s_norm[1])
        x_in.append(col1)
        y_in.append(col2)
    return x_in,y_in


eol_data = np.load('%s/battery_EoL.npy'%(data_dir),allow_pickle='TRUE')
battery_id = np.load('%s/index_battery.npy'%(data_dir),allow_pickle='TRUE')
charge_data=np.load('%s/charge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
discharge_data=np.load('%s/discharge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
summary_data=np.load('%s/summary_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
charge_norm=np.load('%s/charge_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()
discharge_norm=np.load('%s/discharge_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()
summary_norm=np.load('%s/summary_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()
feature_selector_ch=tf.keras.models.load_model('%s/feature_selector_ch.h5'%(model_dir), compile=False)
feature_selector_dis=tf.keras.models.load_model('%s/feature_selector_dis.h5'%(model_dir), compile=False,custom_objects={'mish':mish})
predictor=tf.keras.models.load_model('%s/predictor.h5'%(model_dir), compile=False,custom_objects={'mish':mish})
index=np.load('%s/index_battery.npy'%(data_dir))

cell_feature={}



for i in tqdm(range(len(charge_data))):
    charge_feature=feature_selector(feature_selector_ch,
                                    charge_data[i],charge_norm)
    discharge_feature=feature_selector(feature_selector_dis,
                                    discharge_data[i],discharge_norm)
    cell_feature['%d'%(i)]=concat_data(charge_feature,discharge_feature,
                               summary_data[i])    
s_norm,rul_norm=re_norm(cell_feature)
x_in,y_in=process2predict(cell_feature,s_norm,rul_norm)
tf.keras.backend.clear_session()
in_x1,in_x2=[x_in[i] for i in index[17:]],[x_in[j] for j in index[:17]]
in_x2=np.vstack(in_x2).reshape(-1,50,12)
in_x1=np.vstack(in_x1).reshape(-1,50,12)
in_y1,in_y2=[y_in[i] for i in index[17:]],[y_in[j] for j in index[:17]]
in_y2=np.vstack(in_y2).reshape(-1,2)
in_y1=np.vstack(in_y1).reshape(-1,2)

predict_renorm=np.stack((rul_norm,s_norm)).reshape(2,2)

p1=predictor.predict(in_x1,batch_size=256)*predict_renorm[:,1]+predict_renorm[:,0]
p2=predictor.predict(in_x2,batch_size=256)*predict_renorm[:,1]+predict_renorm[:,0]

ans1=in_y1*predict_renorm[:,1]+predict_renorm[:,0]
ans2=in_y2*predict_renorm[:,1]+predict_renorm[:,0]

print('training_RUL_mae:%.3f'%(mae(p1[:,0],ans1[:,0])))
print('training_S_mae:%.3f'%(mae(p1[:,1],ans1[:,1])))
print('testing_RUL_mae:%.3f'%(mae(p2[:,0],ans2[:,0])))
print('testing_S_rmae:%.3f'%(mae(p2[:,1],ans2[:,1])))
