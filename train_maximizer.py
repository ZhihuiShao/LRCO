import numpy as np
import pandas as pd
import random
from collections import namedtuple, deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization,Flatten,Dense, Concatenate
from tensorflow.python.keras.layers import Input,Activation,Dropout,Reshape,Softmax, Activation
from tensorflow.python.keras import Sequential, models, optimizers, layers, losses, metrics, regularizers
import tensorflow.python.keras.backend as K
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as sop
import math
# global parameters
global_user = 20
global_x_min = 0
global_x_max = 1
global_eta = 0.02
global_epsilon = 0.71

def obj_fun(x,a): # input is numpy [None,20]
    """ object function to minimize """
    x,a = x.reshape([-1,global_user]),a.reshape([-1,global_user])
    res = 1-x*a  #[None,20]
    res = res.reshape([-1,4,5]) #[None,4,5]
    res = 1-np.prod(res,axis=-1) #[None,4]
    res = np.prod(res,axis=-1) #[None]
    res = res-global_eta*np.sum(a,axis=-1) #[None,]
    return res #[None,]

# load training data
x_train = np.tile(sio.loadmat('./data/x_train.mat')['p_linear'],[4,1,1])
x_val = np.tile(sio.loadmat('./data/x_val.mat')['p_linear'],[4,1,1])
x_test = np.tile(sio.loadmat('./data/x_test.mat')['p_linear'],[4,1,1])

# load all actions
def load_actions(name='val'):
    action_val1 = sio.loadmat('./result/'+name+'_random_actions.mat')['action']
    action_val2 = sio.loadmat('./result/'+name+'_greedy_actions.mat')['action']
    action_val3 = sio.loadmat('./result/'+name+'_random_actions_v1.mat')['action']
    action_val4 = sio.loadmat('./result/'+name+'_optimal_actions.mat')['action']
    action_val = np.concatenate((action_val1,action_val2,action_val3,action_val4),axis=0)#action_val3
    return action_val

action_train = load_actions(name='train')
action_val = load_actions(name='val')
action_test = load_actions(name='test')

def get_maxi_model(hiddens,model_name):
    """maxi models """
    xa = Input(shape=(global_user*2,),name='input')
    a = xa[:,global_user:]
    h = Dense(hiddens[0],activation=None,name='hidden_0')(xa)
    h = Activation(activation='relu',name='Act_0')(h)
    for i,h_num in enumerate(hiddens[1:]):
        h = Dense(h_num,activation='relu',name='hidden_{}'.format(i+1))(h)
        h = Activation(activation='relu',name='Act_{}'.format(i+1))(h)
    out = Dense(units=global_user,activation='tanh',name='out')(h)#noise
    out = 0.71*out*a
    model = models.Model(inputs=xa,outputs=out,name=model_name)
    return model

x_train_pair = np.hstack((x_train.reshape([-1,global_user]),action_train.reshape([-1,global_user])))
print(x_train_pair.shape)

# pre-trained for different lambda
for lamb in [1,2,5,8,10]:
    print(lamb,':\t','./checkpoints_pretrain_new/max_model_layer2_200_lam{}'.format(lamb))
    def my_max_loss(y_true, y_pred): #[None,300],None[150]
        y_true_x = y_true[:,:global_user]+y_pred
        y_true_a = y_true[:,global_user:]
        y_true_x = tf.reshape(y_true_x,[-1,4,5])
        y_true_a = tf.reshape(y_true_a,[-1,4,5])
        xa = 1-(y_true_x)*y_true_a #shape=(None,4,5)
        loss = K.prod(1-K.prod(xa,axis=-1),axis=-1,keepdims=True) # shape=(None,1)
        loss = loss-global_eta*K.sum(K.sum(y_true_a,axis=-1),axis=-1,keepdims=True) # shape(None,1)
        bounds = K.sqrt(K.sum(K.square(y_pred),axis=-1,keepdims=True))-global_epsilon # shape: (None,1)
        bounds1 = K.sum(K.sum(K.maximum(1.0,y_true_x)-1,axis=-1),axis=-1,keepdims=True)
        bounds2 = K.sum(K.sum(K.maximum(0.0,-1*y_true_x),axis=-1),axis=-1,keepdims=True)
        loss2 = K.maximum(bounds,0) # shape: (None,1)
        return loss+lamb*loss2+10*(bounds1+bounds2)
    maxi_model = get_maxi_model(hiddens=[400,400],model_name='maxi')
    maxi_model.compile(loss=my_max_loss,optimizer=optimizers.Adam(learning_rate=1e-3))
    maxi_model.fit(x_train_pair,x_train_pair,batch_size=128,epochs=50,verbose=1)
    maxi_model.save_weights('./checkpoints_pretrain/worst_case_model_layer2_400_lam{}'.format(lamb))