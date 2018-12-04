import tensorflow as tf
from model.utils.preprocessing_data_forBNN import FuzzyMultivariateTimeseriesBNNUber
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
link = './data/google_trace_timeseries/data_resource_usage_5Minutes_6176858948.csv'
colnames = ['cpu_rate','mem_usage','disk_io_time','disk_space'] 
df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[3,4,9,10], engine='python')
scaler = MinMaxScaler(feature_range=(0, 1))
cpu = df['cpu_rate'].values.reshape(-1,1)
mem = df['mem_usage'].values.reshape(-1,1)
disk_io_time = df['disk_io_time'].values.reshape(-1,1)
disk_space = df['disk_space'].values.reshape(-1,1)

link_fuzzy = './data/fuzzied/5minutes_ver3.csv'
fuzzy_df = read_csv(link, header=None, index_col=False, names=colnames, usecols=[0,1,2,3], engine='python')
fuzzied_cpu = fuzzy_df['cpu_rate'].values.reshape(-1,1)
fuzzied_mem = fuzzy_df['mem_usage'].values.reshape(-1,1)
fuzzied_disk_io_time = fuzzy_df['disk_io_time'].values.reshape(-1,1)
fuzzied_disk_space = fuzzy_df['disk_space'].values.reshape(-1,1)
original_data = [cpu,mem]
prediction_data = [cpu]
external_feature = [cpu]

train_size = int(0.6 * len(cpu))
# print (train_size)
valid_size = int(0.2 * len(cpu))
sliding_encoder = 18
sliding_decoder = 2
sliding_inference = 8
input_dim = len(original_data)
number_out_decoder = 1
timeseries = FuzzyMultivariateTimeseriesBNNUber(original_data, prediction_data, external_feature, train_size, valid_size, sliding_encoder, sliding_decoder, sliding_inference, input_dim,number_out_decoder)
train_x_encoder, valid_x_encoder, test_x_encoder, train_x_decoder, valid_x_decoder, test_x_decoder, train_y_decoder, valid_y_decoder, test_y_decoder, min_y, max_y, train_x_inference, valid_x_inference, test_x_inference, train_y_inference, valid_y_inference, test_y_inference = timeseries.prepare_data()

train_x_encoder = np.array(train_x_encoder)
train_x_decoder = np.array(train_x_decoder)
test_x_encoder = np.array(test_x_encoder)
test_x_decoder = np.array(test_x_decoder)
test_y_decoder = np.array(test_y_decoder)
train_x_inference = np.array(train_x_inference)
# print ('train_x_inference')
# print (train_x_inference)
test_x_inference = np.array(test_x_inference)
test_y_inference = np.array(test_y_inference)
print (test_x_inference.shape)
# lol
folder = './results/fuzzy/multivariate/cpu/5minutes/bnn_multivariate_uber_ver8/model_saved/18-2-8-4-16_4-1-2-2-16-1-0.9/'
link = folder + 'model.meta'
tf.reset_default_graph()
sess = tf.Session()
# def load_model():
    #First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(link)
saver.restore(sess,tf.train.latest_checkpoint(folder))
# Access and create placeholders variables and
graph = tf.get_default_graph()
file = open('lol.txt','a+', encoding="utf8")
for n in tf.get_default_graph().as_graph_def().node:
    file.write(n.name + '\n')
x_encoder = graph.get_tensor_by_name("x_encoder:0")
x_inference = graph.get_tensor_by_name("x_inference:0")
output_layer = graph.get_tensor_by_name("output_layer/Sigmoid:0")
prediction = sess.run(output_layer, feed_dict={x_encoder:test_x_encoder, x_inference:test_x_inference})
print (prediction.shape)
prediction = prediction * (max_y[0] - min_y[0]) + min_y[0]
err = tf.reduce_mean(tf.abs(tf.subtract(prediction,test_y_inference)) )
error = sess.run(err)
print (error)