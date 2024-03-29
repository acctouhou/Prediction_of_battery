
# Deep Neural Network Battery Life and Voltage Prediction by Using Data of One Cycle Only

## Introduction

This repository contains our dataset, pre-trained model, and predicting script of 'Battery Life and Voltage Prediction by Using Data of One Cycle Only.' Two experiments illustrated following set up our concept of data flow and the design of models. 

Pipeline:
```
|-----Cycling charging data--------feature_selector_ch--|-----predictor-----Prediction RUL & used cycle
|-----Cycling discharging data-----feature_selector_dis-|-----predictor2----Prediction voltage v.s. SOC curve & capacity & power entire life
|-----Summarizing data per cycle------------------------|
```

## Datasets

### I. Description

The dataset is pre-processing from the Severson et al. work. After data cleaning, we split the feature into charge, discharge, and summary parts. The roughly visualization of the dataset represent in data_visualization.ipynb. Then, the inferring of a cell and all of the cells demonstrated in inferring.ipynb and predict.py, respectively.  


### II. Demo of data processing

The Jupyter notebook(4_data processing/data_processing.ipynb) performs how the raw data has been processed. The processing includes data cleaning, interpolating, and collecting from MAT-files.


### III. Trained model& dataset 
We have released weight and dataset for the main models in the paper.
 
Download link:
https://drive.google.com/drive/folders/1XTG8GaSqbpFglACoJ61-0Slg2MKeVCtA?usp=share_link

### IV. Battery Remaining Useful Life and Voltage Prediction
Make sure the folder architecture as follows:
```
|-----dataset-----|----charge_data.npy  
|                 |----discharge_data.npy
|                 |----summary_data.npy
|                 |----battery_???.npy (bags of target ex.EoL)
|                 |----???_norm.npy (bags of formulas of standardization function)
|
|----pretrained---|----feature_selector_ch.h5
|                 |----feature_selector_dis.h5
|                 |----predictor.h5 (for RUL)
|                 |----predictor2.h5 (for voltage)
|
|----predict.py (predicting all sample)
|----data_visualization.ipynb
|----inferring_???.ipynb (predicting a sample)
```
Environment:
```
Python==3.7.6 
tensorflow-gpu==2.2.0
scikit-learn==0.22.2.post1
tqdm

GPU RTX2080Ti
RAM 128G
```
## Predicting

predict.py  

This script demonstrates how feature selector and predictor work. Then the model performance in the training and testing set is evaluated as well.

inferring_???.ipynb  

The notebook shows how to inferring a cell with a specific battery, start, and appending data.

For example  

```
Battery_id      9
Used          100 cycles
Append_cycles   5 cycles
```

In inferring_RUL.ipynb  
```
Predict RUL/S is 611/82 cycle
Ground truth RUL/S is 609/100 cycles
```
In inferring_voltage.ipynb  

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/1_Predicting/figure_voltage.PNG)
(a) Predict the voltage v.s. SOC curve at SOH=90% with above setting  (b) Predict the voltage v.s. SOC curve at SOH=80~100% with above setting

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/1_Predicting/figure_capacity.PNG)
(a) Predict the capacity of whole battery life with above setting  (b) Predict the power of whole battery life with above setting



## Experiments in Supplementary

### I. How the last padding leads into time-series regularization?

In our research, we design the neural network to assist the gradient descent algorithm fitting linear curves. The trajectory of optimization would be the feature of our model. With the last padding technique, Our method can force the neural network aggregation the varying length of sequences. Thus, it could represent high accuracy in the early stage of the feature, which regards as time-series regularization.

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/2_Experiment1/figure1.png)
(a)Benchmark of our method in ANN & CNN (b)Show different length of data guide different stages of training processing

### II. How the undegradate battery repersent their feature?

We introduce the NLP technique to deal with forecasting battery life. After tokenizing the battery feature, the data is fed into ALBERT (unsupervised language representation learning algorithm) and compares the latent information between non-degradation and low-degradation.

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/3_Experiment2/figure2.png)
(a)Show the vocabulary size(Precision of measurement) could affect the distinguishable of degradation (b)illustrated the distance between Nth cycle and 100th cycle feature in ALBERT latent space




## Citing

### BibTeX

```
@article{HSU2022118134,
title = {Deep neural network battery life and voltage prediction by using data of one cycle only},
journal = {Applied Energy},
volume = {306},
pages = {118134},
year = {2022},
issn = {0306-2619},
doi = {https://doi.org/10.1016/j.apenergy.2021.118134},
```
## Licensing

This repository is licensed under the Apache License, Version 2.0. See LICENSE for the full license text.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=acctouhou/Prediction_of_battery&type=Date)](https://star-history.com/#acctouhou/Prediction_of_battery&Date)
