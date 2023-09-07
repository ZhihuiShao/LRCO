import numpy as np
import pandas as pd
import random
from collections import namedtuple, deque
import tensorflow as tf
from tensorflow.distributions import Categorical
from tensorflow import keras
from tensorflow.python.keras.layers import BatchNormalization,Flatten,Dense,Input,Activation,Dropout,Reshape,Softmax, LSTM
from tensorflow.python.keras.layers import Conv2D, Bidirectional, GRU
from tensorflow.python.keras import Sequential, models, optimizers, layers, losses, metrics, regularizers
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.python.keras.backend as K
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as sop
import math
from scipy.optimize import minimize

# global parameters
global_user = 20
global_x_min = 0
global_x_max = 1
global_eta = 0.02
global_epsilon = 0.71

## load the lam=5 maxi
lamb = 1
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
    return loss+lamb*loss2+100*(bounds1+bounds2)
# maximize model
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

tf.reset_default_graph()

hidden_nodes = [200,200]#[200,200]
layer_num = 2
file_path = '../checkpoints_pretrain/worst_case_model_layer'
model_1 = get_maxi_model(hiddens=hidden_nodes,model_name='maxi')
model_1.compile(loss=my_max_loss,optimizer=optimizers.Adam(learning_rate=1e-5))
model_1.load_weights(file_path+'{}_200_lam1'.format(layer_num))
model_2 = get_maxi_model(hiddens=hidden_nodes,model_name='maxi')
model_2.compile(loss=my_max_loss,optimizer=optimizers.Adam(learning_rate=1e-5))
model_2.load_weights(file_path+'{}_200_lam2'.format(layer_num))

model_5 = get_maxi_model(hiddens=hidden_nodes,model_name='maxi')
model_5.compile(loss=my_max_loss,optimizer=optimizers.Adam(learning_rate=1e-5))
model_5.load_weights(file_path+'{}_200_lam5'.format(layer_num))

model_10 = get_maxi_model(hiddens=hidden_nodes,model_name='maxi')
model_10.compile(loss=my_max_loss,optimizer=optimizers.Adam(learning_rate=1e-5))
model_10.load_weights(file_path+'{}_200_lam8'.format(layer_num))


def calcuate_worst_case(x_val_pair):
    x_val = x_val_pair[:,:global_user]
    action_val = x_val_pair[:,global_user:]
    # get worse
    x_val_worse_1 = model_1.predict(x_val_pair).reshape([-1,global_user])
    x_val_worse_2 = model_2.predict(x_val_pair).reshape([-1,global_user])
    x_val_worse_5 = model_5.predict(x_val_pair).reshape([-1,global_user])
    x_val_worse_10 = model_10.predict(x_val_pair).reshape([-1,global_user])

    dx_1 = np.sqrt(np.sum(np.square(x_val_worse_1),axis=-1,keepdims=True))
    dx_2 = np.sqrt(np.sum(np.square(x_val_worse_2),axis=-1,keepdims=True))
    dx_5 = np.sqrt(np.sum(np.square(x_val_worse_5),axis=-1,keepdims=True))
    dx_10 = np.sqrt(np.sum(np.square(x_val_worse_10),axis=-1,keepdims=True))
    ratio_1 = np.maximum(dx_1/global_epsilon,1)
    ratio_2 = np.maximum(dx_2/global_epsilon,1)
    ratio_5 = np.maximum(dx_5/global_epsilon,1)
    ratio_10 = np.maximum(dx_10/global_epsilon,1)

    x_val_worse_add_1 = x_val_pair[:,:global_user]+x_val_worse_1/ratio_1
    x_val_worse_add_2 = x_val_pair[:,:global_user]+x_val_worse_2/ratio_2
    x_val_worse_add_5 = x_val_pair[:,:global_user]+x_val_worse_5/ratio_5
    x_val_worse_add_10 = x_val_pair[:,:global_user]+x_val_worse_10/ratio_10
    x_val_worse_add_1 = np.minimum(np.maximum(x_val_worse_add_1,global_x_min),global_x_max)
    x_val_worse_add_2 = np.minimum(np.maximum(x_val_worse_add_2,global_x_min),global_x_max)
    x_val_worse_add_5 = np.minimum(np.maximum(x_val_worse_add_5,global_x_min),global_x_max)
    x_val_worse_add_10 = np.minimum(np.maximum(x_val_worse_add_10,global_x_min),global_x_max)
    obj_1 = obj_fun(x_val_worse_add_1,action_val).reshape([-1,1])
    obj_2 = obj_fun(x_val_worse_add_2,action_val).reshape([-1,1])
    obj_5 = obj_fun(x_val_worse_add_5,action_val).reshape([-1,1])
    obj_10 = obj_fun(x_val_worse_add_10,action_val).reshape([-1,1])
    obj_ensemble = np.hstack((obj_1,obj_2,obj_5,obj_10))
    obj_worse = np.min(obj_ensemble,axis=-1,keepdims=True)
    return obj_worse


def get_robust_performance(x_val,actions): # return [None,1]
    my_p = calcuate_worst_case(np.hstack((x_val.reshape([-1,global_user]),
                                          actions.reshape([-1,global_user]))))
    return my_p


# load data
x_train = sio.loadmat('../data/x_train.mat')['p_linear']
x_val = sio.loadmat('../data/x_val.mat')['p_linear']
x_test = sio.loadmat('../data/x_test.mat')['p_linear']
x_test_true = sio.loadmat('../data/x_test.mat')['p_true']


def obj_fun(x,a): # input is numpy [None,20]
    """ object function to minimize """
    x,a = x.reshape([-1,global_user]),a.reshape([-1,global_user])
    res = 1-x*a  #[None,20]
    res = res.reshape([-1,4,5]) #[None,4,5]
    res = 1-np.prod(res,axis=-1) #[None,4]
    res = np.prod(res,axis=-1) #[None]
    res = res-global_eta*np.sum(a,axis=-1) #[None,]
    return res #[None,]


hidden_size = [50,50]
global_T = 1


# def the actor graph # begin by result all
tf.random.set_random_seed(0)
random.seed(0)
np.random.seed(0)
# define the graph
x_input = tf.placeholder(dtype=tf.float32,shape=[None,global_user,],name='x_input')
a_value = tf.placeholder(dtype=tf.float32,shape=[None,1],name='value_input')
with tf.variable_scope("actor"):
    h = Reshape(target_shape=(global_user,1,1),name='in_reshape')(x_input)
    embed = Conv2D(filters=hidden_size[0],kernel_size=(1,1),strides=(1,1),activation='relu',name='embed')(h)
    embed = Reshape(target_shape=(global_user,hidden_size[0]),name='in_reshape')(embed)
    h1 = LSTM(hidden_size[1], return_sequences=True, return_state=False,name='LSTM1')(embed)
    logit = LSTM(1, return_sequences=True, return_state=False,name='logit')(h1) #[None,10,1]
    logit = Reshape(target_shape=(global_user,),name='out_reshape')(logit)#[None,10]
    logit = logit/global_T
    out_prob = Activation(activation='sigmoid',name='output')(logit)#[None,10]
    out_prob = Reshape(target_shape=(global_user,1))(out_prob)
    out_prob_r = 1-out_prob
    out_prob_final = tf.concat((out_prob,out_prob_r),axis=-1)#[None,10,2]
    # obtain probability by sampling and get the onehot_p as the final action
    dist = Categorical(probs=out_prob_final)
    action = dist.sample()
    action = tf.cast(action,tf.int32) #shape=(None,10)
    select_action = tf.placeholder(dtype=tf.int32,shape=[None,global_user],name='select_action')
    log_p = dist.log_prob(select_action) #shape=(None,15) corresponding to the log of position
    log_p_sum = tf.reduce_sum(log_p,axis=-1,keep_dims=True)
    reward = tf.placeholder(dtype=tf.float32,shape=[None,1],name='reward')
    # get the loss and optimizer
    delta_reward = reward-a_value # shape(None,1)
    actor_loss = tf.reduce_mean(-1*delta_reward*log_p_sum,axis=0) # shape(0,)
    # set for lr decay
    global_step1 = tf.Variable(0, trainable=False, name="global_step1")
    lr1_start = 2e-3 # initial learning rate
    lr1_decay_rate = 0.9 # learning rate decay rate
    lr1_decay_step = 20*10 # learning rate decay step
    lr1 = tf.train.exponential_decay(lr1_start,global_step1,lr1_decay_step,
                                     lr1_decay_rate, staircase=False, name="learning_rate1")
    actor_opt = tf.train.AdamOptimizer(learning_rate=lr1,beta1=0.9,beta2=0.99, epsilon=0.0000001)   
    gvs = actor_opt.compute_gradients(actor_loss)
    capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]
    actor_train = actor_opt.apply_gradients(capped_gvs,global_step=global_step1) #actor_train = actor_opt.minimize(actor_loss)
saver = tf.train.Saver(max_to_keep=2)


init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
avg_rewar_list = []
avg_log_p_sum_list = []
train_size = 1000
for i in range(100):
    # train actor
    for _ in range(10): #train 10 times
        index = np.random.choice(15000,train_size)
        actor_x_input = x_train[index]
        # predict action
        predict_action = sess.run(action,{x_input:actor_x_input}) #[None,10]
        predict_reward = get_robust_performance(actor_x_input,predict_action) #[None,1]
        # 1. simulate the avg reward
        avg_reward = np.zeros((train_size,0)) #[None,0]
        for _ in range(20):
            onepass_action = sess.run(action,{x_input:actor_x_input}) #[None,5]
            onepass_reward = get_robust_performance(actor_x_input,onepass_action) #[None,1]
            avg_reward = np.hstack((avg_reward,onepass_reward))
        avg_reward = np.mean(avg_reward,axis=-1,keepdims=True) #(None,1)
        # 2. train actor
        sess.run(actor_train,{x_input:actor_x_input,select_action:predict_action,
                              reward:predict_reward,a_value:avg_reward})
    saver.save(sess, './LSTM/' + 'model_2layer_50.ckpt', global_step=i+1)
    if i%1==0:
        cur_action = sess.run(action,{x_input:x_val})
        cur_reward = get_robust_performance(x_val,cur_action)
        cur_log_p_sum,cur_lr1 = sess.run([log_p_sum,lr1],
                                         {x_input:x_val,
                                          select_action:cur_action})
        print(np.mean(cur_reward),np.mean(np.exp(cur_log_p_sum)),cur_lr1)
        avg_rewar_list.append(np.mean(cur_reward))
        avg_log_p_sum_list.append(np.mean(np.exp(cur_log_p_sum)))
