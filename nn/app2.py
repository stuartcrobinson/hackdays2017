# 
# 
# #!flask/bin/python
# # -*- coding: utf-8 -*-
# 
# from flask import Flask
# from flask import request
# import sys
# import numpy
# from keras.models import Sequential
# from keras.layers import Activation
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils
# import datetime
# 
# import tensorflow as tf
# import numpy as np
# 
# import os
# import os.path as osp
# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io
# from keras.models import load_model
# from keras import backend as K
# 
# '''
# #start elasticsearch:
# sudo -u bronto sh /usr/local/bronto/commerce-es/bin/elasticsearch -d
# #start solr
# ~/solr/solr-7.1.0/bin/solr start -c -p 8983 -s example/cloud/node1/solr
# ~/solr/solr-7.1.0/bin/solr start -c -p 7574 -s example/cloud/node2/solr -z localhost:9983
# '''
# 
# 
# 
# # Create function to convert saved keras model to tensorflow graph
# def convert_to_pb(h5_model_file, input_fld='',output_fld=''):
#    
#     # h5_model_file is a .h5 keras model file
#     output_node_names_of_input_network = ["pred0"] 
#     output_node_names_of_final_network = 'output_node'
#     
#     # change filename to a .pb tensorflow file
#     output_graph_name = h5_model_file[:-2]+'pb'                  
#     weight_file_path = osp.join(input_fld, h5_model_file)
#     
#     net_model = load_model(weight_file_path)
#     
#     num_output = len(output_node_names_of_input_network)
#     pred = [None]*num_output
#     pred_node_names = [None]*num_output
#     
#     for i in range(num_output):
#         pred_node_names[i] = output_node_names_of_final_network+str(i)
#         pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
#         
#     sess = K.get_session()
#     
#     constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
#     graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
#     print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))
#     
#     return output_fld+output_graph_name
# 
# 
# def load_graph(frozen_graph_filename):
#     # We load the protobuf file from the disk and parse it to retrieve the 
#     # unserialized graph_def
#     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     
#     # Then, we can use again a convenient built-in function to import a graph_def into the 
#     # current default Graph
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(
#             graph_def, 
#             input_map=None, 
#             return_elements=None, 
#             name="prefix", 
#             op_dict=None, 
#             producer_op_list=None
#         )
#     
#     input_name = graph.get_operations()[0].name+':0'
#     output_name = graph.get_operations()[-1].name+':0'
#     
#     return graph, input_name, output_name
# 
# def predict(model_path, input_data):
#     # load tf graph
#     tf_model,tf_input,tf_output = load_graph(model_path)
#     
#     # Create tensors for model input and output
#     x = tf_model.get_tensor_by_name(tf_input)
#     y = tf_model.get_tensor_by_name(tf_output) 
#     
#     # Number of model outputs
#     num_outputs = y.shape.as_list()[0]
#     predictions = np.zeros((input_data.shape[0],num_outputs))
#     for i in range(input_data.shape[0]):        
#         with tf.Session(graph=tf_model) as sess:
#             y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
#             predictions[i] = y_out
#     
#     return predictions
# 
# def predictFromTfModel(tf_model, tf_input, tf_output, input_data):
#     # load tf graph
# #     tf_model,tf_input,tf_output = load_graph(model_path)
#     
#     # Create tensors for model input and output
#     x = tf_model.get_tensor_by_name(tf_input)
#     y = tf_model.get_tensor_by_name(tf_output) 
#     
#     # Number of model outputs
#     num_outputs = y.shape.as_list()[0]
#     predictions = np.zeros((input_data.shape[0],num_outputs))
#     for i in range(input_data.shape[0]):        
#         with tf.Session(graph=tf_model) as sess:
#             y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
#             predictions[i] = y_out
#     
#     return predictions
# 
# 
# 
# siteId ='35569'# '38178' #'35569'
# 
# #https://stackoverflow.com/questions/44274701/make-predictions-using-a-tensorflow-graph-from-a-keras-model
# keras_model_file = '~/ml/model_dir/ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2.h5'
# 
# map_productId_index = numpy.load('~/ml/dataMedium/map_productId_index_'+siteId+'.npy').item()
# map_index_productId = dict((v,k) for (k,v) in map_productId_index.items())
# map_productId_imgUrl = numpy.load('~/ml/dataMedium/map_productId_imgUrl_'+siteId+'.npy').item()
# map_productId_parentProductId = numpy.load('~/ml/dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()
# map_parentProductId_productId = dict((parentProductId, productId) for productId, parentProductId in map_productId_parentProductId.items())
# 
# tf_model_path = convert_to_pb(keras_model_file,'~/ml/model_dir/','~/ml/model_dir2/')
# tf_model,tf_input,tf_output = load_graph(tf_model_path)
# 
# 
# value = 9
#     
# #!flask/bin/python
# from flask import Flask
# 
# app = Flask(__name__)
# 
# @app.route('/test')
# def index2():
# 
# #     print("summary in test:")# 
# #     model.summary()
#     
#     productIdsNew = ['OP-MKT10-PORIII-ET-K','OP-MKT12-PORIII-K','OP-MKT6510AT-HBG']
#     Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
#     for productId in productIdsNew:
#         productIndex = map_productId_index[productId]
#         Xnew[0][productIndex] = 1
# #     prediction = model.predict(Xnew, verbose=0)[0]
#     prediction = predictFromTfModel(tf_model,tf_input,tf_output, Xnew)[0]
#     
#     
#     for productId in productIdsNew:
#         productIndex = map_productId_index[productId]
#         prediction[productIndex] = 0
#     next_productIndex = numpy.argmax(prediction)
#     next_productId = map_index_productId[next_productIndex]
#     return "hi " + str(value) + " ok " + str(next_productId)
# 
# 
# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0', port=5001)    
# 
