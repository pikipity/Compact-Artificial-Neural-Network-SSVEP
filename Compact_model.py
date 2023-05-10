#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

keras.backend.clear_session()

class ecca_feature_extraction_layer(layers.Layer):
    def __init__(self, target_num, channelNum, timeLen, harmonicNum, 
                       filterNum, unit=1,
                       name='ecca_feature_extraction_layer', **kwargs):
        super(ecca_feature_extraction_layer, self).__init__(name=name+'_subband'+str(int(filterNum)), **kwargs)
        self.target_num = target_num
        self.channelNum = channelNum
        self.timeLen = timeLen
        self.harmonicNum = harmonicNum
        self.filterNum = filterNum
        self.unit = unit
        
    def call(self, inputs, sine, template):
        feature_sine = tf.matmul(inputs, sine, transpose_b=(True))
        feature_template = tf.matmul(inputs, template, transpose_b=(True))
        signal_en = tf.matmul(inputs, inputs, transpose_b=(True))
        sine_en = tf.matmul(sine, sine, transpose_b=(True))
        template_en = tf.matmul(template, template, transpose_b=(True))
        return feature_sine, feature_template, signal_en, sine_en, template_en
    def get_config(self):
        config = super(ecca_feature_extraction_layer, self).get_config()
        config.update({"target_num": self.target_num,
                    "channelNum": self.channelNum,
                    "timeLen": self.timeLen,
                    "harmonicNum": self.harmonicNum,
                    "filterNum": self.filterNum,
                    "unit": self.unit})
        return config

class ecca_corr_layer_0(layers.Layer):
    def __init__(self, channelNum, harmonicNum, filterNum, unit_w=1, unit_v=1,
                       name='ecca_corr_layer_0', **kwargs):
        super(ecca_corr_layer_0, self).__init__(name=name+'_subband'+str(int(filterNum)))
        self.channelNum = channelNum
        self.harmonicNum = harmonicNum
        self.filterNum = filterNum
        self.unit_w = unit_w
        self.unit_v = unit_v
        
        self.weight_w = self.add_weight(name='ecca_u0_subband'+str(int(filterNum)),
                                        shape=(unit_w, channelNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
        self.weight_v = self.add_weight(name='ecca_v0_subband'+str(int(filterNum)),
                                        shape=(unit_w, harmonicNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
    def call(self, feature_sine, signal_en, sine_en):
        features_1 = tf.matmul(self.weight_w, feature_sine)
        features_1 = tf.matmul(features_1, self.weight_v, transpose_b=True)
        signal_en = tf.matmul(self.weight_w, signal_en)
        signal_en = tf.matmul(signal_en, self.weight_w, transpose_b=True)
        sine_en = tf.matmul(self.weight_v, sine_en)
        sine_en = tf.matmul(sine_en, self.weight_v, transpose_b=True)
        r0 = tf.divide(features_1,
                       tf.sqrt(tf.matmul(signal_en,sine_en)))
        return r0
    def get_config(self):
        config = super(ecca_corr_layer_0, self).get_config()
        config.update({"channelNum": self.channelNum,
                        "harmonicNum": self.harmonicNum,
                        "filterNum": self.filterNum,
                        "unit_w": self.unit_w,
                        "unit_v": self.unit_v})
        return config
    
class ecca_corr_layer_1(layers.Layer):
    def __init__(self, channelNum, harmonicNum, filterNum, unit_w=1, unit_v=1,
                       name='ecca_corr_layer_1', **kwargs):
        super(ecca_corr_layer_1, self).__init__(name=name+'_subband'+str(int(filterNum)))
        self.channelNum = channelNum
        self.harmonicNum = harmonicNum
        self.filterNum = filterNum
        self.unit_w = unit_w
        self.unit_v = unit_v
        
        self.weight_w = self.add_weight(name='ecca_u1_subband'+str(int(filterNum)),
                                        shape=(unit_w, channelNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
        self.weight_v = self.add_weight(name='ecca_v1_subband'+str(int(filterNum)),
                                        shape=(unit_w, harmonicNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
    def call(self, feature_sine, feature_template, signal_en, sine_en, template_en):
        features_1 = tf.matmul(self.weight_w, feature_sine)
        features_1 = tf.matmul(features_1, self.weight_v, transpose_b=True)
        features_2 = tf.matmul(self.weight_w, feature_template)
        features_2 = tf.matmul(features_2, self.weight_w, transpose_b=True)
        signal_en = tf.matmul(self.weight_w, signal_en)
        signal_en = tf.matmul(signal_en, self.weight_w, transpose_b=True)
        sine_en = tf.matmul(self.weight_v, sine_en)
        sine_en = tf.matmul(sine_en, self.weight_v, transpose_b=True)
        template_en = tf.matmul(self.weight_w, template_en)
        template_en = tf.matmul(template_en, self.weight_w, transpose_b=True)
        r1 = tf.divide(features_1,
                       tf.sqrt(tf.matmul(signal_en,sine_en)))
        r2 = tf.divide(features_2,
                       tf.sqrt(tf.matmul(signal_en,template_en)))
        return r1, r2
    def get_config(self):
        config = super(ecca_corr_layer_1, self).get_config()
        config.update({"channelNum": self.channelNum,
                        "harmonicNum": self.harmonicNum,
                        "filterNum": self.filterNum,
                        "unit_w": self.unit_w,
                        "unit_v": self.unit_v})
        return config

class ecca_corr_layer_2(layers.Layer):
    def __init__(self, channelNum, harmonicNum, filterNum, unit_w=1, unit_v=1,
                       name='ecca_corr_layer_2', **kwargs):
        super(ecca_corr_layer_2, self).__init__(name=name+'_subband'+str(int(filterNum)))
        self.channelNum = channelNum
        self.harmonicNum = harmonicNum
        self.filterNum = filterNum
        self.unit_w = unit_w
        self.unit_v = unit_v
        
        self.weight_w = self.add_weight(name='ecca_u2_subband'+str(int(filterNum)),
                                        shape=(unit_w, channelNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
        self.weight_v = self.add_weight(name='ecca_v2_subband'+str(int(filterNum)),
                                        shape=(unit_w, channelNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
    def call(self, feature_sine, feature_template, signal_en, sine_en, template_en):
        features_2 = tf.matmul(self.weight_w, feature_template)
        features_2 = tf.matmul(features_2, self.weight_v, transpose_b=True)
        signal_en = tf.matmul(self.weight_w, signal_en)
        signal_en = tf.matmul(signal_en, self.weight_w, transpose_b=True)
        template_en = tf.matmul(self.weight_v, template_en)
        template_en = tf.matmul(template_en, self.weight_v, transpose_b=True)
        r2 = tf.divide(features_2,
                       tf.sqrt(tf.matmul(signal_en,template_en)))
        return r2
    def get_config(self):
        config = super(ecca_corr_layer_2, self).get_config()
        config.update({"channelNum": self.channelNum,
                        "harmonicNum": self.harmonicNum,
                        "filterNum": self.filterNum,
                        "unit_w": self.unit_w,
                        "unit_v": self.unit_v})
        return config
    
class ecca_corr_layer_3(layers.Layer):
    def __init__(self, channelNum, harmonicNum, filterNum, unit_w=1, unit_v=1,
                       name='ecca_corr_layer_3', **kwargs):
        super(ecca_corr_layer_3, self).__init__(name=name+'_subband'+str(int(filterNum)))
        self.channelNum = channelNum
        self.harmonicNum = harmonicNum
        self.filterNum = filterNum
        self.unit_w = unit_w
        self.unit_v = unit_v
        
        self.weight_w = self.add_weight(name='ecca_u3_subband'+str(int(filterNum)),
                                        shape=(unit_w, channelNum),
                                        initializer = keras.initializers.RandomNormal(),
                                        trainable = True)
    def call(self, feature_sine, feature_template, signal_en, sine_en, template_en):
        features_2 = tf.matmul(self.weight_w, feature_template)
        features_2 = tf.matmul(features_2, self.weight_w, transpose_b=True)
        signal_en = tf.matmul(self.weight_w, signal_en)
        signal_en = tf.matmul(signal_en, self.weight_w, transpose_b=True)
        template_en = tf.matmul(self.weight_w, template_en)
        template_en = tf.matmul(template_en, self.weight_w, transpose_b=True)
        r2 = tf.divide(features_2,
                       tf.sqrt(tf.matmul(signal_en,template_en)))
        return r2
    def get_config(self):
        config = super(ecca_corr_layer_3, self).get_config()
        config.update({"channelNum": self.channelNum,
                        "harmonicNum": self.harmonicNum,
                        "filterNum": self.filterNum,
                        "unit_w": self.unit_w,
                        "unit_v": self.unit_v})
        return config

class filterbankCombineLayer(layers.Layer):
    def __init__(self, filterNum, name = 'filterbankCombineLayer', **kwargs):
        super(filterbankCombineLayer, self).__init__(name=name, **kwargs)
        self.filterNum = filterNum
        
        self.w = self.add_weight(name='filterbank_weight',
                                     shape=(1, filterNum),
                                     initializer=keras.initializers.RandomNormal(),
                                     trainable=True)
    def call(self, inputs):
        out = tf.matmul(self.w, inputs)
        return out
    def get_config(self):
        config = super(filterbankCombineLayer, self).get_config()
        config.update({"filterNum": self.filterNum})
        return config
    
class featureCombineLayer_addweight(layers.Layer):
    def __init__(self, filterNum = 0, name='featureCombineLayer_addweight', **kwargs):
        super(featureCombineLayer_addweight, self).__init__(name=name+'_subband'+str(int(filterNum)), **kwargs)
        self.filterNum = filterNum
        
        self.w = self.add_weight(name='feature_weights_subband'+str(int(filterNum)),
                                 shape=(1, 5),
                                 initializer=keras.initializers.RandomNormal(),
                                 constraint = keras.constraints.UnitNorm(axis=1),
                                 trainable=True)
    def call(self, r0, r1, r2, r3, r4):
        r = layers.concatenate([r0, r1, r2, r3, r4], axis=-2)
        r = tf.math.multiply(tf.math.sign(r), tf.pow(r,2))
        out = tf.matmul(self.w, r)
        return out
    def get_config(self):
        config = super(featureCombineLayer_addweight, self).get_config()
        config.update({"filterNum": self.filterNum})
        return config
    
def gen_ecca_model_filterbank(target_num, filterNum, timeLen, channelNum, harmonicNum, unit=1):
    input_layer = layers.Input(shape=(filterNum, channelNum, timeLen), dtype=tf.float32)
    sine_cosine_layer = layers.Input(shape=(target_num, harmonicNum, timeLen), dtype=tf.float32)
    averaged_template_layer = layers.Input(shape=(filterNum, target_num, channelNum, timeLen), dtype=tf.float32)
    
    featureCombine_layer = featureCombineLayer_addweight()
    filterbank_combine_layer = filterbankCombineLayer(filterNum)
    
    input_layer_exp = tf.expand_dims(input_layer, 2)
    input_layer_exp = tf.repeat(input_layer_exp, target_num, 2)
    
    feature_all=[]
    for j in range(filterNum):
        feature_sine, feature_template, signal_en, sine_en, template_en = ecca_feature_extraction_layer(
            target_num, channelNum, timeLen, harmonicNum, j)(input_layer_exp[:,j,:], sine_cosine_layer, averaged_template_layer[:,j,:])
        r0 = ecca_corr_layer_0(channelNum, harmonicNum, j)(feature_sine, signal_en, sine_en)
        r1, r2 = ecca_corr_layer_1(channelNum, harmonicNum, j)(feature_sine, feature_template, signal_en, sine_en, template_en)
        r3 = ecca_corr_layer_2(channelNum, harmonicNum, j)(feature_sine, feature_template, signal_en, sine_en, template_en)
        r4 = ecca_corr_layer_3(channelNum, harmonicNum, j)(feature_sine, feature_template, signal_en, sine_en, template_en)
        r = featureCombine_layer(r0, r1, r2, r3, r4)
        feature_all.append(r)
    out = layers.concatenate(feature_all, axis=-2)
    out = filterbank_combine_layer(out)
    out = keras.activations.softmax(out, axis=1)
    out = tf.squeeze(out, axis=[-1, -2])
    
    model = Model([input_layer, sine_cosine_layer, averaged_template_layer], out)
    return model
    
    
