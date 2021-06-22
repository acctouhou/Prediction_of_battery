# Deep Neural Network Battery Life and Voltage Prediction by Using Data of One Cycle Only

## Introduction

This repository contains our dataset, pre-trained model, and predicting script of 'Battery Life and Voltage Prediction by Using Data of One Cycle Only.' Two experiments illustrated following set up our concept of data flow and the design of models. 


## Datasets

The dataset is pre-processing from the Severson et al. work. After data cleaning, we split the feature into charge, discharge, and summary parts. The roughly visualization of the dataset represent in data_visualization.ipynb. Then, the inferring of a cell and all of the cells demonstrated in inferring.ipynb and predict.py, respectively.  


### Battery Remaining Useful Life and Voltage Prediction
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




## Experiments in Supplementary

### I. How the last padding leads into time-series regularization?

In our research, we design the neural network to assist the gradient descent algorithm fitting linear curves. The trajectory of optimization would be the feature of our model. With the last padding technique, Our method can force the neural network aggregation the varying length of sequences. Thus, it could represent high accuracy in the early stage of the feature, which regards as time-series regularization.

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/Experiment1/figure1.png)
(a)benchmark of our method in ANN & CNN (b)show different length of data guide different stages of training processing

### II. How the undegradate battery repersent their feature?

We introduce the NLP technique to deal with forecasting battery life. After tokenizing the battery feature, the data is fed into ALBERT (unsupervised language representation learning algorithm) and compares the latent information between non-degradation and low-degradation.

![image](https://github.com/acctouhou/Prediction_of_battery/blob/main/Experiment2/figure2.png)
(a)show the vocabulary size(Precision of measurement) could affect the distinguishable of degradation (b)illustrated the distance between Nth cycle and 100th cycle feature in ALBERT latent space




## Citing

### BibTeX

```
@misc{acctouhou,
  author = {Chia-wei,Hsu},
  title = {Battery-Life-and-Voltage-Prediction},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/acctouhou/Battery-Life-and-Voltage-Prediction}}
}
```

