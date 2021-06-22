# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:06:46 2019

@author: Acc
"""
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer, TFAutoModel

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from tokenizers.trainers import WordPieceTrainer
from transformers import AutoTokenizer, TFAutoModel
import scipy.stats
from scipy.special import softmax


import sys
import os

from sklearn import preprocessing
def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    v=a.var_
    return d,m,v,
def norm2(data):
    a= preprocessing.MinMaxScaler().fit(data.reshape(-1,1))
    d=a.transform(data.reshape(-1,1))
    return d+1e-9
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
data_dir='dataset'

vo=int(sys.argv[1])

vols=[100,
200,
 300,
  400,
   500,
    600,
     700,
      800,
       900,
        1000,
         2000,
          3000,
           4000,
            5000,
             6000,
              7000,
               8000,
                9000,
                 10000,
                  20000]
vo=vols[vo]







x_data=np.load('%s/discharge_data_exp2.npy'%(data_dir),allow_pickle='TRUE')

y_data=np.load('%s/battery_eol_exp2.npy'%(data_dir),allow_pickle='TRUE')


dict_data='vo_%d'%(vo)
x_data=np.transpose(x_data,(0,2,1)).reshape(-1,4)
n_x_data,_,_,_=norm(x_data)


kmeans = MiniBatchKMeans(n_clusters=vo,random_state=0,batch_size=64)

discrete_x_data=kmeans.fit_predict(np.float32(n_x_data))

discrete_x_data=discrete_x_data.reshape(-1,500)
log=[]
for i in tqdm(range(len(discrete_x_data))):
    temp=''
    for j in range(500):
        temp+='%s '%(p1[i,j])
    temp+='.'
    log.append(temp)
    
    

with open(dict_data, 'w') as f:
    for item in tqdm(log):
        f.write("%s\n" % item)




bert_tokenizer = Tokenizer(WordPiece())


bert_tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=vo)
data=[dict_data]
bert_tokenizer.train(trainer, data)


model_name = "albert-xxlarge-v2"
tf_model = TFAutoModel.from_pretrained(model_name)



log1=[]
for i in tqdm(range(115)):
    output = bert_tokenizer.encode(log[i*100+99]).ids
    ref=tf_model.predict(np.array(output[:499]))[0]
    for j in range(100):
        output = bert_tokenizer.encode(log[i*100+j]).ids
        p1=tf_model.predict(np.array(output[:499]))[0]
        log1.append(np.abs(p1-ref).sum())

np.save('compare%d_log'%(vo),np.array(log1))
