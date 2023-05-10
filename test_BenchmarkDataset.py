#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import to_categorical

import numpy as np
import scipy.io
import os.path
import shutil
import random
random.seed(220309)

import timeit
import datetime

from Compact_model import gen_ecca_model_filterbank

def copyNewModel(model_source):
    model_target = keras.models.clone_model(model_source)
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
    return model_target

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

keras.backend.clear_session()

current_learning_rate=1e-3
current_momentum=0.1
optimizer = keras.optimizers.SGD(learning_rate=current_learning_rate, momentum=current_momentum)
loss_metric = keras.metrics.Mean() 
test_loss_metric = keras.metrics.Mean()

fs=250
trial_no=40
block_no=6
sub_no=35
epoch_no=1000

train_block_list = []
test_block_list = []
validation_block_list=[]
for i in range(block_no):
    allBlock = list(range(block_no))
    allBlock.pop(i)
    test_block_list.append([i])
    train_block_list.append(allBlock)
    
timeLen_list = np.floor(np.arange(0.25,5,0.25)*fs)

if os.path.isfile('test_BenchmarkDataset.mat'):
    print('-----------------------')
    print('Load old data')
    tmp=scipy.io.loadmat('test_BenchmarkDataset.mat')
    train_acc_store=tmp['train_acc_store']
    test_acc_store=tmp['test_acc_store']
    loss_history=tmp['loss_history']
    test_loss_history=tmp['test_loss_history']
else:
    print('-----------------------')
    print('Build new data')
    loss_history = np.zeros((len(timeLen_list),
                             len(train_block_list),
                             sub_no,
                             epoch_no,
                             block_no-1))
    
    test_loss_history = np.zeros((len(timeLen_list),
                             len(train_block_list),
                             sub_no,
                             epoch_no,
                             1))
    
    
    train_acc_store = np.zeros((len(timeLen_list),
                          len(train_block_list),
                          sub_no,
                          epoch_no))
    test_acc_store = np.zeros((len(timeLen_list),
                          len(train_block_list),
                          sub_no,
                          epoch_no))
    

for timeLen_i in range(len(timeLen_list)):
    timeLen = int(timeLen_list[timeLen_i])
    for test_run in range(len(train_block_list)):
        for sub in range(sub_no):
            
            fileName = 'sub_'+str(sub+1)+'.mat'
            allData = scipy.io.loadmat(fileName)['data'] # filterbank, trial, block, channel, time
            oneTrialData = tf.constant(allData[0,0,0,:,0:int(timeLen)])
            exact_time_len = oneTrialData.shape[1]
            
            train_block = train_block_list[test_run]
            test_block = test_block_list[test_run]
            
            sineMat = scipy.io.loadmat('sine_ref.mat')['sine_ref']
            
            
            t1_total = timeit.default_timer()
            
            print('--------------------------------------')
            t1 = timeit.default_timer()
            cca_classifier = gen_ecca_model_filterbank(trial_no, 5, timeLen, 9, 10)
            del optimizer, loss_metric, test_loss_metric 
            optimizer = keras.optimizers.SGD(learning_rate=current_learning_rate, momentum=current_momentum)
            loss_metric = keras.metrics.Mean() 
            test_loss_metric = keras.metrics.Mean()
            
            template_tmp = allData[:, 0:int(trial_no), train_block, 0:int(9), 0:int(timeLen)]
            template_tmp = tf.constant(template_tmp, dtype = tf.float32)
            template_tmp = tf.reduce_mean(template_tmp, axis=2)
            template_tmp = tf.expand_dims(template_tmp, 0)
            
            sine_tmp = sineMat[0:int(trial_no), 0:int(10), 0:int(timeLen)]
            sine_tmp = tf.constant(sine_tmp, dtype=tf.float32)
            sine_tmp = tf.expand_dims(sine_tmp, 0)
            
            template_train_store=[]
            for block_i in range(len(train_block)):
                block = train_block[block_i]
                train_block_tmp = train_block.copy()
                train_block_tmp.pop(train_block_tmp.index(block))
                print(train_block_tmp)
                template_train_tmp = allData[:, 0:int(trial_no), train_block_tmp, 0:int(9), 0:int(timeLen)]
                template_train_tmp = tf.constant(template_train_tmp, dtype = tf.float32)
                template_train_tmp = tf.reduce_mean(template_train_tmp, axis=2)
                template_train_tmp = tf.expand_dims(template_train_tmp, 0)
                template_train_store.append(template_train_tmp)
            
            t2 = timeit.default_timer()
            print('Build new model, time: %.4f' % (t2-t1))
            
            for epoch in range(epoch_no):

                disp_message_flag = 0
                if epoch%50==0:
                    disp_message_flag=1
                else:
                    disp_message_flag=0
                
                if disp_message_flag:
                    print('-----------------------------------------')
                    print('Time Index: %s/%s (%s), Run: %s/%s, Sub: %s/%s, epoch: %s/%s' 
                                                                      % (timeLen_i+1,len(timeLen_list),exact_time_len,
                                                                         test_run+1,len(train_block_list),
                                                                         sub+1,sub_no,
                                                                         epoch+1,epoch_no))
                    print('Train blocks: %s, Test blocks: %s, learning rate: %s' % (train_block, test_block, current_learning_rate))
                
                # Train
                t1 = timeit.default_timer()
                for block_i in range(len(train_block)):
                    block = train_block[block_i]
                    
                    randomTrial = list(range(trial_no))
                    random.shuffle(randomTrial)
                    
                    
                    if block_i==0:
                        trainX = allData[:,randomTrial,block,:,0:int(timeLen)]
                        trainX = np.transpose(trainX, (1,0,2,3))

                        trainY = randomTrial
                        trainY = to_categorical(trainY, 40)
                        
                        batchNum = 40
                        sine = tf.repeat(sine_tmp, batchNum, 0)
                        template = tf.repeat(template_train_store[block_i], batchNum, 0)
                    else:
                        trainX_tmp = allData[:,randomTrial,block,:,0:int(timeLen)]
                        trainX_tmp = np.transpose(trainX_tmp, (1,0,2,3))
                        trainX = np.concatenate((trainX, trainX_tmp), axis = 0)

                        trainY_tmp = randomTrial
                        trainY_tmp = to_categorical(trainY_tmp, 40)
                        trainY = tf.concat(((trainY, trainY_tmp)), axis = 0)
                        
                        batchNum = 40
                        tmp = tf.repeat(sine_tmp, batchNum, 0)
                        sine = tf.concat((sine, tmp), axis = 0)
                        tmp = tf.repeat(template_train_store[block_i], batchNum, 0)
                        template = tf.concat((template, tmp), axis = 0)
                

                with tf.GradientTape() as tape:
                    pred = cca_classifier([trainX, sine, template])
                    loss = keras.losses.categorical_crossentropy(trainY, pred)

                grads = tape.gradient(loss, cca_classifier.trainable_weights)
                optimizer.apply_gradients(zip(grads, cca_classifier.trainable_weights))
                loss_metric(loss)
                loss_history[timeLen_i,
                                 test_run,
                                 sub,
                                 epoch,
                                 0] = loss_metric.result().numpy()
                t2 = timeit.default_timer()
                if disp_message_flag:
                    print('      final loss: %s, time: %.4f' % (loss_metric.result().numpy(), t2-t1))
                
                # train acc
                t1 = timeit.default_timer()
                corr_trial_num=0
                total_trial_num=0
                for block_i in range(len(train_block)):
                    block = train_block[block_i]
                    
                    randomTrial = list(range(trial_no))
                    random.shuffle(randomTrial)
                    
                    trainX = allData[:,randomTrial,block,:,0:int(timeLen)]
                    trainX = np.transpose(trainX, (1,0,2,3))
                    
                    trainY = randomTrial
                    
                    batchNum = 40
                    sine = tf.repeat(sine_tmp, batchNum, 0)
                    template = tf.repeat(template_train_store[block_i], batchNum, 0)
                    
                    pred = cca_classifier([trainX, sine, template])
                    pred_label = tf.argmax(pred,axis=1).numpy()
                    
                    for trial in range(trial_no):
                        if abs(pred_label[trial]-trainY[trial])<1e-3:
                            corr_trial_num += 1
                        total_trial_num +=1
                train_acc_store[timeLen_i,
                          test_run,
                          sub,
                          epoch]=corr_trial_num/total_trial_num
                t2 = timeit.default_timer()
                if disp_message_flag:
                    print('      train mean acc: %.2f%%, time: %.4f' % (train_acc_store[timeLen_i,
                                                                      test_run,
                                                                      sub,
                                                                      epoch]*100,t2-t1))
                
                
                # test acc
                t1 = timeit.default_timer()
                corr_trial_num=0
                total_trial_num=0
                for block_i in range(len(test_block)):
                    block = test_block[block_i]
                    
                    randomTrial = list(range(trial_no))
                    random.shuffle(randomTrial)
                    
                    testX = allData[:,randomTrial,block,:,0:int(timeLen)]
                    testX = np.transpose(testX, (1,0,2,3))
                    
                    testY = randomTrial
                    
                    batchNum = 40
                    sine = tf.repeat(sine_tmp, batchNum, 0)
                    template = tf.repeat(template_tmp, batchNum, 0)
                    
                    pred = cca_classifier([testX, sine, template])
                    pred_label = tf.argmax(pred,axis=1).numpy()
                    
                    loss = keras.losses.categorical_crossentropy(to_categorical(testY, 40), pred)
                    test_loss_metric(loss)
                    
                    test_loss_history[timeLen_i,
                                     test_run,
                                     sub,
                                     epoch,
                                     block_i] = test_loss_metric.result().numpy()
                    
                    for trial in range(trial_no):
                        if abs(pred_label[trial]-testY[trial])<1e-3:
                            corr_trial_num += 1
                        total_trial_num +=1
                test_acc_store[timeLen_i,
                          test_run,
                          sub,
                          epoch]=corr_trial_num/total_trial_num
                t2 = timeit.default_timer()
                if disp_message_flag:
                    print('      test mean acc : %.2f%%, time: %.4f, loss: %.4f' % (test_acc_store[timeLen_i,
                                                                                test_run,
                                                                                sub,
                                                                                epoch]*100,t2-t1,
                                                                                test_loss_metric.result().numpy(), ))
               
                
            t2_total = timeit.default_timer()
            print('----------------------------------------')
            print('Time Index: %s/%s (%s), Run: %s/%s, Sub: %s/%s -> time: %s' 
                                                                  % (timeLen_i+1,len(timeLen_list),exact_time_len,
                                                                     test_run+1,len(train_block_list),
                                                                     sub+1,sub_no,
                                                                     datetime.timedelta(seconds=t2_total-t1_total), ))
            
            
            
            
            
            print('Save model and data')
            cca_classifier.save('test_BenchmarkDataset/timelen'+str(timeLen_i)+'_run'+str(test_run)+'_sub'+str(sub)+'_model')
            
            del tape
            del cca_classifier
            tf.keras.backend.clear_session()
            
            store_var = {}
            store_var['train_acc_store']=train_acc_store
            store_var['test_acc_store']=test_acc_store
            store_var['loss_history']=loss_history
            store_var['test_loss_history']=test_loss_history
            scipy.io.savemat('test_BenchmarkDataset.mat',store_var)
     