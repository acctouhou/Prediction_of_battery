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
        v=a.var_
        return d,m,s
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
    log1=np.float32(norm(np.array(log1).reshape(-1,1)))[1:]
    log2=np.float32(norm(np.array(log2).reshape(-1,1)))[1:]
    return log1,log2


def process2predict(cell_feature):
    scan_step=10
    aggregation_step=5
    x_in1=[]
    x_in2=[]
    y_in1=[]
    y_in2=[]
    y_in3=[]
    for i in tqdm(range(len(cell_feature))):
        col1=[]
        col2=[]
        col3=[]
        col4=[]
        col5=[]
        len_=len(cell_feature['%d'%(i)])-50  
        
        for k in range(0,len_,scan_step):
            for j in range(0,50,aggregation_step):
                temp=cell_feature['%d'%(i)][k:(j+k+1)]
                for stage in range(100):
                    col1.append(np.float16(np.pad(temp, ((0,50-j-1),(0,0)), 'edge')))
                    col2.append(np.float16(section[i,stage]))
                    col3.append(np.float16(battery_voltage[i,stage]))
                    col4.append(np.float16(battery_capacity[i,stage]))
                    col5.append(np.float16(battery_power[i,stage]))
                
                
        x_in1.append(col1)
        x_in2.append(col2)
        y_in1.append(col3)
        y_in2.append(col4)
        y_in3.append(col5)
    return x_in1,x_in2,y_in1,y_in2,y_in3

battery_voltage = np.load('%s/battery_voltage.npy'%(data_dir),allow_pickle='TRUE')
battery_capacity = np.load('%s/battery_capacity.npy'%(data_dir),allow_pickle='TRUE')
battery_power = np.load('%s/battery_power.npy'%(data_dir),allow_pickle='TRUE')
charge_data=np.load('%s/charge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
discharge_data=np.load('%s/discharge_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
summary_data=np.load('%s/summary_data.npy'%(data_dir),allow_pickle='TRUE').tolist()
charge_norm=np.load('%s/charge_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()
discharge_norm=np.load('%s/discharge_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()
summary_norm=np.load('%s/summary_norm.npy'%(data_dir),allow_pickle='TRUE').tolist()



feature_selector_ch=tf.keras.models.load_model('%s/feature_selector_ch.h5'%(model_dir), compile=False)
feature_selector_dis=tf.keras.models.load_model('%s/feature_selector_dis.h5'%(model_dir), compile=False,custom_objects={'mish':mish})
predictor=tf.keras.models.load_model('%s/predictor2.h5'%(model_dir), compile=False,custom_objects={'mish':mish})

battery_voltage,voltage_mean,voltage_sacle=norm(battery_voltage.reshape(-1,100))
battery_capacity,capacity_mean,capacity_scale=norm(battery_capacity.reshape(-1,1))
battery_power,power_mean,power_scale=norm(battery_power.reshape(-1,1))

section=np.ones([len(battery_voltage),100])
for i in range(100):
    section[:,i]=(i+1)/100
section,section_mean,section_scale=norm(section.reshape(-1,1))
#%%
section=section.reshape(-1,100)
battery_capacity=battery_capacity.reshape(-1,100)
battery_power=battery_power.reshape(-1,100)
battery_voltage=battery_voltage.reshape(-1,100,100)

#%%




cell_feature={}



for i in tqdm(range(len(charge_data))):
    charge_feature=feature_selector(feature_selector_ch,
                                    charge_data[i],charge_norm)
    discharge_feature=feature_selector(feature_selector_dis,
                                    discharge_data[i],discharge_norm)
    cell_feature['%d'%(i)]=concat_data(charge_feature,discharge_feature,
                               summary_data[i])    
x_in1,x_in2,y_in1,y_in2,y_in3=process2predict(cell_feature)


#%%



tf.keras.backend.clear_session()

cell_number=9
phase=99
start_cycle=600   
append=5

in_x1=np.vstack(x_in1[cell_number]).reshape(-1,50,12)
in_x2=np.vstack(x_in2[cell_number]).reshape(-1,1)


predict_voltage,predict_capacity,predict_power=predictor.predict([in_x1,in_x2],batch_size=256)
predict_voltage=predict_voltage.reshape(-1,5,100,100)[start_cycle//10,append//5]
predict_power=predict_power.reshape(-1,5,100)[start_cycle//10,append//5]

predict_capacity=predict_capacity.reshape(-1,5,100)[start_cycle//10,append//5]

#%%
def clip(x):
    return np.clip(x,2,3.6)
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,4),dpi=200)

axes[0].plot(clip(predict_voltage[phase]*voltage_sacle+voltage_mean),label='Prediction',c='b')
axes[0].plot(battery_voltage[cell_number,phase]*voltage_sacle+voltage_mean,label='Ground truth'
         ,ls='--',c='r')
axes[0].legend()
axes[0].set_ylabel('Voltage(V)')
axes[0].set_xlabel('SOC(%)')
axes[1].set_xlabel('SOC(%)')

cm=plt.get_cmap('coolwarm')
for i in range(100):
    axes[1].plot(clip(predict_voltage[i]*voltage_sacle+voltage_mean),c=cm(i/100))
axes[0].set_title('Prediction on SoH=90%')
axes[1].set_title('Continuous prediction')

plt.tight_layout() 

#%%
#%%

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,4),dpi=200)
target_eol=len(x_in1[cell_number])/5/100*10

axes[0].plot(np.linspace(1,target_eol,100),predict_capacity*capacity_scale+capacity_mean,c='b'
             ,label='Prediction')
axes[0].plot(np.linspace(1,target_eol,100),battery_capacity[cell_number]*capacity_scale+capacity_mean,
             ls='--',c='r',label='Ground truth')


axes[1].plot(np.linspace(1,target_eol,100),predict_power*power_scale+power_mean,c='b'
             ,label='Prediction')
axes[1].plot(np.linspace(1,target_eol,100),battery_power[cell_number]*power_scale+power_mean,
             ls='--',c='r',label='Ground truth')
axes[0].set_title('Capacity prediction')
axes[1].set_title('Power prediction')
axes[0].set_xlabel('Cycles')
axes[1].set_xlabel('Cycles')

axes[0].set_ylabel('Capacity(Ah)')
axes[1].set_ylabel('Power(Wh)')
axes[0].legend()
axes[1].legend()

