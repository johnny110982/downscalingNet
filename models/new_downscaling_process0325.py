# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 14:20:14 2017

@author: chang
"""

import tensorflow as tf
import csv
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from numpy import *
from datetime import datetime

#define static variables
TIME_STEP = 300
DOWNSCALING = 0
SAVE_NETVAL = 1
READ_netVal = 1
CONTINUED = 1

LEARNING_RATE2 = 1e-4
WEIGHT_INITIAL = 1e-4
REGULARIZATION = 0
COMPARE_THE_INPUTS = False

ERAI_SHAPE = [29,17]
ERAI_LENGTH = ERAI_SHAPE[0]*ERAI_SHAPE[1]

WRF_SHAPE = [199,119]
WRF_LENGTH = WRF_SHAPE[0]*WRF_SHAPE[1]

name = 'erai_NewData'

print('--------------setting--------------')
print('TIME_STEP = ' + str(TIME_STEP))
print('DOWNSCALING = ' + str(DOWNSCALING))
print('READ_netVal = ' + str(READ_netVal))
print('LEARNING_RATE2 = ' + str(LEARNING_RATE2))
print('WEIGHT_INITIAL = ' + str(WEIGHT_INITIAL))
print('REGULARIZATION = ' + str(REGULARIZATION))
print('-----------------------------------')

#functions
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=WEIGHT_INITIAL)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

def next_batch(erai, wrf):
    print('next_batch...')
    n=3651
    samples=np.random.randint(0,erai.shape[0],n)
    batch_erai = erai[samples]
    batch_wrf = wrf[samples]
    return batch_erai, batch_wrf

'''
    print('-----test-----')
    print('samples = ')
    print(samples.shape)
    print(samples)
    print('batch erai = ')
    print(batch_erai.shape)
    print(batch_erai)
    print('batch wrf = ')
    print(batch_wrf.shape)
    print(batch_wrf)
    print('return')
'''    


#read training data
print('reading train erai')
ds = xr.open_dataset('D:trainERAI1995to2005.nc')
er = ds.tp
erai_train = np.array(er).reshape(-1,ERAI_LENGTH)
#erai_train = erai_train.reshape(-1,2, ERAI_LENGTH)
#erai_train = np.mean(erai_train, axis=1)
print (erai_train.shape)


#read testing data
print('reading test erai')
ds = xr.open_dataset('D:testERAI2005to2015.nc')
er = ds.tp
erai_test = np.array(er).reshape(-1, ERAI_LENGTH)
#erai_test = erai_test.reshape(-1,2, ERAI_LENGTH)
#erai_test = np.mean(erai_test, axis=1)[0:4017-3653,:]
print (erai_test.shape)

#read fake wrf data
print('reading train wrf')
ds = xr.open_dataset('D:trainWRF1995to2005.nc')
er = ds.incrain
wrf_train = np.array(er).reshape(-1,WRF_LENGTH)
print (wrf_train.shape)


print('reading test wrf')
ds = xr.open_dataset('D:testWRF2005to2015.nc')
er = ds.incrain
wrf_test = np.array(er).reshape(-1,WRF_LENGTH)
print (wrf_test.shape)

#wrf_train = np.random.rand(7306, WRF_LENGTH).reshape(-1,WRF_LENGTH)
#wrf_test = np.random.rand(7306, WRF_LENGTH).reshape(-1,WRF_LENGTH)


#data standardization
print('data standardization: getting mean and std')
erai_train_mean = erai_train.mean()
erai_train_std = erai_train.std()
erai_test_mean = erai_test.mean()
erai_test_std = erai_test.std()

wrf_train_mean = wrf_train.mean()
wrf_train_std = wrf_train.std()
wrf_test_mean = wrf_test.mean()
wrf_test_std = wrf_test.std()
print('done')
print('standardizing')
erai_train = (erai_train - erai_train_mean)/erai_train_std
erai_test = (erai_test - erai_test_mean)/erai_test_std
wrf_train = (wrf_train - wrf_train_mean)/wrf_train_std
wrf_test = (wrf_test - wrf_test_mean)/wrf_test_std
print('done')
#erai_train_mean


#Building network
print('building network')
graph = tf.Graph()
with graph.as_default():
    x1 = tf.placeholder("float", shape=[None,ERAI_LENGTH])
    y = tf.placeholder("float", shape=[None,WRF_LENGTH])
    x1_image = tf.reshape(x1,[-1,ERAI_SHAPE[0], ERAI_SHAPE[1],1])
#    y_image = tf.reshape(y, [-1,WRF_SHAPE[0], WRF_SHAPE[1]])
    W_conv1_x1 = weight_variables([7,7,1,48])
    b_conv1_x1 = bias_variables([48])
    h_conv1_x1 = tf.nn.relu(conv2d(x1_image, W_conv1_x1) + b_conv1_x1)
    h_pool1_x1 = max_pool_3x3(h_conv1_x1)
    
    W_conv2_x1 = weight_variables([5,5,48,64])
    b_conv2_x1 = bias_variables([64])
    h_conv2_x1 = tf.nn.relu(conv2d(h_pool1_x1, W_conv2_x1) + b_conv2_x1)
    h_pool2_x1 = max_pool_3x3(h_conv2_x1)
    h_pool2_flat_x1 = tf.reshape(h_pool2_x1, [-1,8*5*64])

    W_fc1_x1 = weight_variables([8*5*64,1024]) 
    b_fc1_x1 = bias_variables([1024])
    h_fc1_x1 = tf.nn.relu(tf.matmul(h_pool2_flat_x1, W_fc1_x1) + b_fc1_x1)
    
    W_fc2 = weight_variables([1024,WRF_LENGTH])
    b_fc2 = bias_variables([WRF_LENGTH])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_x1, W_fc2) + b_fc2)
    
    x_out = h_fc2
    
    #setting training
    if REGULARIZATION == 1:
        print("regularization = tf.nn.l2_loss(W_fc1)")
        lamda = 1e-3
        print("lamda = " + str(lamda))
        loss = tf.reduce_mean(tf.square(x_out - y)) + lamda*tf.nn.l2_loss(W_fc2)
        mse = tf.reduce_mean(tf.square(x_out - y))
    else:
        loss = tf.reduce_mean(tf.square(x_out - y))
    
    #assinging
    if READ_netVal == 1:
        wconv1_x1 = tf.placeholder("float", shape=[7,7,1,48])
        bconv1_x1 = tf.placeholder("float", shape=[48])
        wconv2_x1 = tf.placeholder("float", shape=[5,5,48,64])
        bconv2_x1 = tf.placeholder("float", shape=[64])
        wfc1_x1 = tf.placeholder("float", shape=[8*5*64,1024])
        bfc1_x1 = tf.placeholder("float", shape=[1024])
        wfc2 = tf.placeholder("float", shape=[1024, WRF_LENGTH])
        bfc2 = tf.placeholder("float", shape=[WRF_LENGTH])
        
        set_wconv1_x1 = W_conv1_x1.assign(wconv1_x1)
        set_bconv1_x1 = b_conv1_x1.assign(bconv1_x1)
        set_wconv2_x1 = W_conv2_x1.assign(wconv2_x1)
        set_bconv2_x1 = b_conv2_x1.assign(bconv2_x1)
        set_wfc1_x1 = W_fc1_x1.assign(wfc1_x1)
        set_bfc1_x1 = b_fc1_x1.assign(bfc1_x1)
        set_wfc2 = W_fc2.assign(wfc2)
        set_bfc2 = b_fc2.assign(bfc2)
        
    #gradient descent optimizer
    train_step2 = tf.train.AdamOptimizer(LEARNING_RATE2).minimize(loss)
        
    print('read network')
    #session start
    print('Sess = tf.Session()')
    sess = tf.Session(graph=graph)
    print('sess.run(tf.initialize_all_variables())')
    sess.run(tf.global_variables_initializer())
    print('done')
    

    #downscaling
    if DOWNSCALING == 1:
        print('start downscaling')
        t0 = datetime.now()
        print(t0)    
        print('train data downscaling')
        t_size = erai_train.shape[0]    #erai_train.shape[0]=493
        print('t_size = ' + str(t_size))
        tEnd = 0
        N = 10
        mse = np.zeros([2,N])
        for i in range(N):
            print(i)
            tStart = tEnd
            tEnd = int(t_size/N) * (i+1)
            if tEnd > t_size:
                tEnd = t_size
            batch_erai = erai_train[tStart:tEnd,:]
            batch_wrf = wrf_train[tStart:tEnd,:]
            result = sess.run(x_out, feed_dict={x1:batch_erai})
            print(result[0].shape)
            print('calculating mse')
            ss=0
            for j in range(tEnd-tStart):
                ss += ((result[0][j] - wrf_train[tStart+j])**2).mean()
                ss = ss/(tEnd-tStart)
                #print('mse = ', ss)
                #print('rmse = ', sqrt(ss))
            print('saving mse')
            mse[0,i] = ss
            mse[1,i] = tStart - tEnd
            print('saving output')
            result[0].astype(float32).tofile('D:data/output/ERAItoWRF_DS_ConvNet_standardized_'+str(TIME_STEP)+'step_train_'+name+'_%02d'%i+'.dat')
    
        print('done')
        print('rmse for all train data is...')
        ss=0
        for i in range(N):
            ss += mse[0,i]*mse[1,i]
        ss = ss/sum(mse[1])
        print(np.sqrt(ss))
    
        print('test data downscaling')
        t_size = erai_test.shape[0]
        print('t_size = ' + str(t_size))
        output = np.zeros([t_size,WRF_LENGTH])
        tEnd = 0
        N = 10
        for i in range(N):
            print(i)
            tStart = tEnd
            tEnd = int(t_size/N) * (i+1)
            if tEnd > t_size:
                tEnd = t_size
            batch_erai = erai_test[tStart:tEnd,:]
            result = sess.run(x_out, feed_dict={x1:batch_erai})
            output[tStart:tEnd,:] = result[0]
        #print('writing output')
        #output.astype(float32).tofile('D:ERAItoWRF_DS_ConvNet_standatdized'+str(TIME_STEP)+'step_test_'+name+'.dat')
    
        
        print('calculating mse')
        ss=0
        for i in range(t_size):
            ss += ((output[i] - wrf_test[i])**2).mean()
        ss = ss/t_size
        '''
        print('mse = ')
        print(mse)
        '''
        print('rmse = ')
        print(np.sqrt(ss))
    
        t1 = datetime.now()
        print('all time is ')
        print(t1-t0)
        print('END')
        #exit()
        print('Downscaling complete!')        
    
    #csv writer
    if REGULARIZATION == 1:
        save_csv = np.zeros((TIME_STEP,2))
    else:
        save_csv = np.zeros((TIME_STEP,1))    
    
    
    if READ_netVal == 1:
        print('read netVals')
        t1 = datetime.now()
        print(t1)
        if CONTINUED == 1:
            print('read CONTINUED netVal')
            Wconv1_x1 = np.fromfile('D:netVals/ConvNet_W_conv1_x1_'+name+'.dat',float32).reshape(7,7,1,48)
            Bconv1_x1 = np.fromfile('D:netVals/ConvNet_b_conv1_x1_'+name+'.dat',float32).reshape(48)
            Wconv2_x1 = np.fromfile('D:netVals/ConvNet_W_conv2_x1_'+name+'.dat',float32).reshape(5,5,48,64)
            Bconv2_x1 = np.fromfile('D:netVals/ConvNet_b_conv2_x1_'+name+'.dat',float32).reshape(64)
            Wfc1_x1 = np.fromfile('D:netVals/ConvNet_W_fc1_x1_'+name+'.dat',float32).reshape(8*5*64,1024)
            Bfc1_x1 = np.fromfile('D:netVals/ConvNet_b_fc1_x1_'+name+'.dat',float32).reshape(1024)
            Wfc2 = np.fromfile('D:netVals/ConvNet_W_fc2_x1_'+name+'.dat',float32).reshape(1024,WRF_LENGTH)
            Bfc2 = np.fromfile('D:netVals/ConvNet_b_fc2_x1_'+name+'.dat',float32).reshape(WRF_LENGTH)
            print('done')
            print('assinging value')
            sess.run(set_wconv1_x1, feed_dict={wconv1_x1: Wconv1_x1})
            sess.run(set_bconv1_x1, feed_dict={bconv1_x1: Bconv1_x1})
            sess.run(set_wconv2_x1, feed_dict={wconv2_x1: Wconv2_x1})
            sess.run(set_bconv2_x1, feed_dict={bconv2_x1: Bconv2_x1})
            sess.run(set_wfc1_x1, feed_dict={wfc1_x1: Wfc1_x1})
            sess.run(set_bfc1_x1, feed_dict={bfc1_x1: Bfc1_x1})
            sess.run(set_wfc2, feed_dict={wfc2: Wfc2})
            sess.run(set_bfc2, feed_dict={bfc2: Bfc2})
            print('done')
            t2 = datetime.now()
            print(t2)
            print('The time is ')
            print(t2-t1)
    
    #train loop
    if DOWNSCALING == 0:
        print('start training')
        t0 = datetime.now()
        print(t0)
        t1 = t0
        for i in range(TIME_STEP):
            print('step ' + str(i))
            t2 = datetime.now()
#            print(t2)
#            print('step time is '+print(t2-t1))
            print('all time is ' + str(t2-t0))
            t1=t2
            batch_erai, batch_wrf = next_batch(erai_train, wrf_train)
        
            if REGULARIZATION == 1:
                result = sess.run([mse,loss], feed_dict={x1:batch_erai, y:batch_wrf})
                print('After_Net: Mean Squared Error =')
                print(result[0])
                print('loss =')
                print(result[1])
                save_csv[i] = result
            else:
#                print("FOR DEBUGGING............")
#                result = sess.run([x1,y,x1_image, h_conv1_x1, h_pool1_x1, h_conv2_x1,h_pool2_x1, h_pool2_flat_x1], feed_dict={x1:batch_erai, y:batch_wrf})
#                for i in range(8):
#                    print(result[i].shape)
                result = sess.run([loss],feed_dict={x1:batch_erai, y:batch_wrf})
                print('After_Net: Mean Squared Error ='+str(result[0]) )
            save_csv[i,0] = result[0]
            print("train")
            sess.run(train_step2, feed_dict={ x1:batch_erai, y:batch_wrf })   
            #saving
            if (i+1)%100 == 0:
                print("saving start")
                f = open('D:output/ERAItoWRF_ConvNet_standardized_'+str(TIME_STEP)+'step_erai_new.csv','w')
                writer = csv.writer(f, lineterminator='\n')
                if REGULARIZATION == 1:
                    for j in range(i+1):
                        writer.writerow([j+1,save_csv[j,0],save_csv[j,1]])
                else:
                    for j in range(i+1):
                        writer.writerow([j+1,save_csv[j,0]])
                f.close()
            
            #saving netVal
            if (i+1)%10 == 0:
                print("netVal saving")
                if SAVE_NETVAL == 1:
                    netVals = sess.run([ W_conv1_x1,b_conv1_x1,W_conv2_x1,b_conv2_x1,  W_fc1_x1,b_fc1_x1, W_fc2, b_fc2])
                    netVals[0].astype(float32).tofile('D:netVals/ConvNet_W_conv1_x1_'+name+'.dat')
                    netVals[1].astype(float32).tofile('D:netVals/ConvNet_b_conv1_x1_'+name+'.dat')
                    netVals[2].astype(float32).tofile('D:netVals/ConvNet_W_conv2_x1_'+name+'.dat')
                    netVals[3].astype(float32).tofile('D:netVals/ConvNet_b_conv2_x1_'+name+'.dat')
                    netVals[4].astype(float32).tofile('D:netVals/ConvNet_W_fc1_x1_'+name+'.dat')
                    netVals[5].astype(float32).tofile('D:netVals/ConvNet_b_fc1_x1_'+name+'.dat')
                    netVals[6].astype(float32).tofile('D:netVals/ConvNet_W_fc2_x1_'+name+'.dat')
                    netVals[7].astype(float32).tofile('D:netVals/ConvNet_b_fc2_x1_'+name+'.dat')
                print('netVal saving done')
    
    
            
    IMAGE_OUTPUT = 1
    if IMAGE_OUTPUT == 1:
        output = sess.run([x1,x_out,y,], feed_dict={x1:batch_erai, y:batch_wrf})
        output[0].astype(float32).tofile('D:output/ConvNet_erai_input.dat')
        output[1].astype(float32).tofile('D:output/ConvNet_dplg_output.dat')
        output[2].astype(float32).tofile('D:output/ConvNet_wrf_output.dat')
        print('Finish IMAGE_OUTPUT')
    
    t2 = datetime.now()
    print('The time is')
    print(t2)
    print('All time is')
    print(t2-t0)
    print('END')

    
    
