import tensorflow as tf
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow.compat.v1 as tfcf
import time

tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_v2_behavior()

tf.random.set_seed(13)

config=tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.visible_device_list='0'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.reset_default_graph()

df = pd.read_csv('data/USDT_BTC.csv')
df.columns = ['date','high','low','open','close','volume','quoteVolume','weightedAverage']
df['ema'] = df.close.ewm(span=14).mean()

def condJ(i, endIndex):
    return i < endIndex

def cond(i, startIndex, endIndex):
    return (i > startIndex) and (j < endIndex)

def bodyJ(j, step, out, data):
    j = j + step
    out = tf.concat([out, data], 0)
    return [j, out]

def bodyI(i, data, labels, dataset, history_size, step, single_step, target_size, target):
    i = i + 1
    j = tf.constant(i-history_size)
    _, data = tf.while_loop(cond(i-history_size, i), bodyJ(step, data, dataset[j]), [j, data], shape_invariants=[j.get_shape(), tf.TensorShape([None])])
    if single_step:
        labels = tf.concat([labels, target[i+target_size]],0)
    else:
       _, data = tf.while_loop(cond(i, i+target_size), bodyJ(1, labels, target[j]), [j, labels], shape_invariants=[j.get_shape(), tf.TensorShape([None])]
    return [i, data, labels]

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size
  start_time = time.time()
  with tf.compat.v1.Session(config=config) as session:
    i = tf.constant(start_index) 
    data = tf.Variable([])
    labels = tf.Variable([])
    _, data, labels = tf.while_loop(cond(start_index, end_index), 
                                  bodyI(data, labels, dataset, history_size, step, single_step, target_size, target), 
                                  [i, data, labels], shape_invariants=[i.get_shape(), tf.TensorShape([None])])
    session.run(tf.compat.v1.global_variables_initializer())
    total_time = time.time() - start_time
    print('Total time of splitting on GPU: ', total_time)
    session.close()
  return data, labels
 
features_considered = ['open', 'close', 'ema']
features = df[features_considered]
features.index = df['date']

TRAIN_SPLIT = 300000
BATCH_SIZE = 256
BUFFER_SIZE = 10000
STEP = 1
EVALUATION_INTERVAL = 200
EPOCHS = 10
future_target = 36
past_history = 720

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

x_train_multi, y_train_multi = multivariate_data(dataset1, dataset[:, 1], 0,
                                                 TRAIN_SPL, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPL, None, past_history,
                                             future_target1, STEP)
sess_eval = tf.compat.v1.Session(config=config)
with sess_eval.as_default():
  x_train_multi = tf.constant(x_train_multi).eval()
  y_train_multi = tf.constant(y_train_multi).eval()
  x_val_multi = tf.constant(x_val_multi).eval()
  y_val_multi = tf.constant(y_val_multi).eval()
  sess_eval.close()

print ('Single window of past history : {}'.format(x_train_multi1[0].shape))
print ('\n Target EMA to predict : {}'.format(y_train_multi1[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
val_shape = x_train_multi.shape[-2:]

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

with tf.compat.v1.Session(config=config) as sess:
   i = tf.constant(0)
   tf.keras.backend.set_session(sess)
   start_time = time.time()
   multi_step_history = multi_step_model.fit(train_data_multi1, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi1 ,
                                          validation_steps=50)
   total_time = time.time() - start_time
   print('Total time of fitting on GPU: ', total_time)
   sess.run(tf.compat.v1.global_variables_initializer())
   sess.close

print("PREDICTED RATES: ")
for x, y in val_data_multi.take(3):
    print(multi_step_model.predict(x)[0])




