#!flask/bin/python
from flask import Flask
from flask import request
import sys
import numpy
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import datetime
# 
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.utils.data_utils import get_file


import tensorflow as tf
import numpy as np

import os
import os.path as osp
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K




# Create function to convert saved keras model to tensorflow graph
def convert_to_pb(h5_model_file, input_fld='',output_fld=''):
   
    # h5_model_file is a .h5 keras model file
    output_node_names_of_input_network = ["pred0"] 
    output_node_names_of_final_network = 'output_node'
    
    # change filename to a .pb tensorflow file
    output_graph_name = h5_model_file[:-2]+'pb'                  
    weight_file_path = osp.join(input_fld, h5_model_file)
    
    net_model = load_model(weight_file_path)
    
    num_output = len(output_node_names_of_input_network)
    pred = [None]*num_output
    pred_node_names = [None]*num_output
    
    for i in range(num_output):
        pred_node_names[i] = output_node_names_of_final_network+str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
        
    sess = K.get_session()
    
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
    print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))
    
    return output_fld+output_graph_name


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    
    input_name = graph.get_operations()[0].name+':0'
    output_name = graph.get_operations()[-1].name+':0'
    
    return graph, input_name, output_name

def predict(model_path, input_data):
    # load tf graph
    tf_model,tf_input,tf_output = load_graph(model_path)
    
    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output) 
    
    # Number of model outputs
    num_outputs = y.shape.as_list()[0]
    predictions = np.zeros((input_data.shape[0],num_outputs))
    for i in range(input_data.shape[0]):        
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
            predictions[i] = y_out
    
    return predictions

def predictFromTfModel(tf_model, tf_input, tf_output, input_data):
    # load tf graph
#     tf_model,tf_input,tf_output = load_graph(model_path)
    
    # Create tensors for model input and output
    x = tf_model.get_tensor_by_name(tf_input)
    y = tf_model.get_tensor_by_name(tf_output) 
    
    # Number of model outputs
    num_outputs = y.shape.as_list()[0]
    predictions = np.zeros((input_data.shape[0],num_outputs))
    for i in range(input_data.shape[0]):        
        with tf.Session(graph=tf_model) as sess:
            y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
            predictions[i] = y_out
    
    return predictions




siteId ='35569'# '38178' #'35569'
# siteId ='38178'
#OP-PEBS2-CHR-K

map_customerId_productIdDateTuplesList = numpy.load('dataMedium/map_customerId_productIdDateTuplesList_'+siteId+'.npy').item()
# map_customerId_productIdDateTuplesList = numpy.load('dataMedium/map_newCustomerId_productIdDateTuplesList_'+siteId+'.npy').item()


map_productId_index = numpy.load('dataMedium/map_productId_index_'+siteId+'.npy').item()
map_index_productId = dict((v,k) for (k,v) in map_productId_index.items())
map_productId_imgUrl = numpy.load('dataMedium/map_productId_imgUrl_'+siteId+'.npy').item()
map_productId_parentProductId = numpy.load('dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()
map_parentProductId_productId = dict((parentProductId, productId) for productId, parentProductId in map_productId_parentProductId.items())

if siteId=='35569':
    keras_model_file = 'ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2.h5'
if siteId=='38178':
    keras_model_file = ''




tf_model_path = convert_to_pb(keras_model_file,'./model_dir/','./model_dir2/')
tf_model,tf_input,tf_output = load_graph(tf_model_path)



def getReqs_ff():
    if siteId == '35569':
        weights = "ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2"# "ffWeights1sloss14.2441_dropout0.2_1hidLayer_35569"# "ffWeights35569"#weights/weightsMedium-improvement-197-2.4338.hdf5"#weightsMedium-improvement-07-6.7973.hdf5" #weightsMedium-improvement-01-5.6892.hdf5" #"weights/weightsMedium-improvement-62-1.4764.hdf5" #weightsMedium-improvement-50-1.4660.hdf5"
        model = Sequential()
        model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
#         model.add(Dropout(0.2))
        model.add(Dense(len(map_productId_index), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
        model.load_weights(weights)
        return model
    else:
        weights = "ffWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_forTestTrain_1800sPauseChops_USDR00253_35569_epoch2"
        model = Sequential()
        model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
        model.add(Dense(len(map_productId_index), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
        model.load_weights(weights)
        return model

def getReqs_lstm():
    if siteId == '35569':
        weights = "weights/GoTweights_Nov29_3_lstm_childIds0.6401"
        model = Sequential()
        model.add(LSTM(len(words), return_sequences=True, input_shape=(maxlen, len(words)))) #was LSTM(512
        model.add(Dropout(0.5)) #was 0.2
        model.add(LSTM(len(words), return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(len(words)))
        model.add(Activation('softmax')) 
    return{'weights': weights,'model': model} 

def getReqs_lstmparent():
    if siteId == '35569':
        weights = "weights/GoTweights_Nov29_1"
        model = Sequential()
        model.add(LSTM(len(words), return_sequences=True, input_shape=(maxlen, len(words)))) #was LSTM(512
        model.add(Dropout(0.5)) #was 0.2
        model.add(LSTM(len(words), return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(len(words)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    else:
        weights = "weights/GoTweights_Nov29_4_lstm_childIds1024_1.3586"
        model = Sequential()
        model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(words)))) #was LSTM(512
        model.add(Dropout(0.5)) #was 0.2
        model.add(LSTM(1024, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(len(words)))
        model.add(Activation('softmax'))
    return{'weights': weights,'model': model}

# map_customerId_productIdList = numpy.load('dataMedium/map_customerId_productIdList_'+siteId+'.npy').item()

# numpy.save('dataMedium/map_customerId_productIdList.npy', map_customerId_productIdList)
# numpy.save('dataMedium/map_customerId_productIdDateTuplesList.npy', map_customerId_productIdDateTuplesList)

# 
# 
# 
# weights = "ffWeightsUsingCountsloss14.4224_dropout0.2_1hidLayer_35569"# "ffWeights1sloss14.2441_dropout0.2_1hidLayer_35569"# "ffWeights35569"#weights/weightsMedium-improvement-197-2.4338.hdf5"#weightsMedium-improvement-07-6.7973.hdf5" #weightsMedium-improvement-01-5.6892.hdf5" #"weights/weightsMedium-improvement-62-1.4764.hdf5" #weightsMedium-improvement-50-1.4660.hdf5"
# model = Sequential()
# model.add(Dense(len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(len(map_productId_index), activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
# 


def baseline_model_ff():
    model = Sequential()
    model.add(Dense(len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(map_productId_index), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
    return model

maxlen=10

def baseline_model_lstm_parentIds():
    words = set(map_productId_parentProductId.values())
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(words))))
    model.add(Dropout(0.5))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

# def baseline_model_lstm(map_productId_index):
#     model = Sequential()
#     model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(map_productId_index))))
#     model.add(Dropout(0.2))
#     model.add(LSTM(512, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(len(map_productId_index)))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#     return model

def baseline_model_lstm(map_productId_index):
    words = set(map_productId_parentProductId.values())
    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, len(words))))
    model.add(Dropout(0.5))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model
    

def sample(preds, diversity=1.0):
    if diversity == 99:
        return numpy.argmax(preds)
    # helper function to sample an index from a probability array
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / diversity
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
#     return numpy.argmax(probas)
    return numpy.argmax(preds)


def getRecommendation_lstm(sentenceInput, model, diversity=1.0):  
    sentence = sentenceInput[-maxlen:]
    x = numpy.zeros((1, maxlen, len(map_productId_index)))
    for t, word in enumerate(sentence):
        x[0, t, map_productId_index[word]] = 1.
    preds = model.predict(x, verbose=0)[0]
# #    don't recommend input products #idk if this even works.  maybe screwing everything up
    for id in sentence:
        index = map_productId_index[id]
        preds[index] = 0
    next_index = sample(preds, diversity)
    next_word = map_index_productId[next_index]
    return next_word


def getRecommendation_lstm_parent(sentenceInput, model, diversity=1.0):  
    words = set(map_productId_parentProductId.values()) #very inefficient 
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    sentence = sentenceInput[-maxlen:] #-maxlen here to take last "maxlen" items in sentence cos that's all model can read.  so there will be repeats of items more than maxlen items ago
    x = numpy.zeros((1, maxlen, len(words)))
    for t, word in enumerate(sentence):
        x[0, t, word_indices[word]] = 1.
    preds = model.predict(x, verbose=0)[0]
# #    don't recommend input products #idk if this even works.  maybe screwing everything up
    for id in sentence:
        index = word_indices[id]
        preds[index] = 0
    next_index = sample(preds, diversity)
    next_word = indices_word[next_index]
    return next_word

def getRec_ff(productIdsNew, model, diversity=99):
    Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
    for productId in productIdsNew:
        if productId not in map_productId_index:
            return "failed: productId not found in browse data: " + productId
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
#     prediction = model.predict(Xnew, verbose=1)[0]
    prediction = predictFromTfModel(tf_model,tf_input,tf_output, Xnew)[0]
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    
    next_productIndex = numpy.argmax(prediction)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs_ff(model, productIdsInput, n=20, diversity=99):
    productIds = productIdsInput[:]
    for i in range(0, n):
        rec = getRec_ff(productIds, model, diversity)
        if "failed" in rec:
            return rec
        productIds.append(getRec_ff(productIds, model, diversity))
    return productIds
    
def getRecs_lstm(weightsFileName, model, productIdsInput, n=20, diversity=1.0):
    productIds = productIdsInput[:]
    model.load_weights(weightsFileName)
    for i in range(0, n):
        productIds.append(getRecommendation_lstm(productIds, model, diversity))
    return productIds

def getRecs_lstm_parent(weightsFileName, model, productIdsInput, n=10, diversity=1.0):
    productIds = productIdsInput[:]
    print("loading weights")
    model.load_weights(weightsFileName)
    print("loaded weights")
    for i in range(0, n):
        productIds.append(getRecommendation_lstm_parent(productIds, model, diversity))
    return productIds


map_index_productId = {}
for k in map_productId_index:
    map_index_productId[map_productId_index[k]] = k
print(len(map_productId_index))


    
#!flask/bin/python
from flask import Flask

app = Flask(__name__)


@app.route('/recommendations/lstmparentids')
def index0():
    print('/recommendations/lstmparentids')
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    diversity = float(request.args.get('diversity', default = '1.0', type = str))
    print(productIdsNew)
    parentProductIds = [map_productId_parentProductId[id] for id in productIdsNew]
    print("got parentProductIds")
    reqs = getReqs_lstmparent()
    result = getRecs_lstm_parent(reqs['weights'], reqs['model'], parentProductIds)
    resultChildProductIds = [map_parentProductId_productId[id] for id in result]
    returner = str(productIdsNew) + "</br>"
    returner = str(parentProductIds) + "</br>"
    returner += "input</br>"
    for productId in productIdsNew:
        returner += "<img src='"+ (map_productId_imgUrl[productId] if productId in map_productId_imgUrl else 'not found') + "' width='100' />"
    returner += "</br>recommended:</br>"
    for productId in resultChildProductIds:
        if productId not in map_productId_imgUrl:
            continue
        returner += "<img src='"+ map_productId_imgUrl[productId] + "' width='100' />"+ productId+"</br>"
    return returner


@app.route('/recommendations/lstm')
def index1():
    print('/recommendations/lstm')
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    diversity = float(request.args.get('diversity', default = '1.0', type = str))
    print(productIdsNew)
    reqs = getReqs_lstm()
    result = getRecs_lstm(reqs['weights'], reqs['model'], productIdsNew)
    returner = str(productIdsNew) + "</br>"
    returner += "input</br>"
    for productId in productIdsNew:
        returner += "<img src='"+ (map_productId_imgUrl[productId] if productId in map_productId_imgUrl else 'not found') + "' width='100' />"
    returner += "</br>recommended:</br>"
    for productId in result:
        if productId not in map_productId_imgUrl:
            continue
        returner += "<img src='"+ map_productId_imgUrl[productId] + "' width='100' />"+ productId+"</br>"
    return returner


# 
weights2 = "ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2"
model2 = Sequential()
model2.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
model2.add(Dense(len(map_productId_index), activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    
model2.load_weights(weights2)

def getReqs_ff2():
    return model2


@app.route('/recommendations/ff')
def index2():
    print("len(map_productId_index)")
    print(len(map_productId_index))
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    model = "" #getReqs_ff()
    result = getRecs_ff(model, productIdsNew)
    if "failed" in result:
        return result
    returner = "Hello, World! " + str(productIdsNew) + "\n"
    returner += "browsed: <br/>\n"
    
    for productId in productIdsNew:
        print("productId")
        print(productId)
        url = map_productId_imgUrl[productId]
        print("url")
        print(url)
        returner += "<img src='"+ url + "' width='200' /> "
    
    returner += "<br/>recommended: "+ str(result) + "<br/>\n"

    print(map_productId_imgUrl['OP-PESS4-CNS-SLT-K'])
    print(result)
    for productId in result:
        if productId not in map_productId_imgUrl:
            continue
        print("productId")
        print(productId)
        url = map_productId_imgUrl[productId]
        print("url")
        print(url)
        returner += "<img src='"+ url + "' width='100' />"+ productId+"</br>"
    return returner
 #    
# 

@app.route('/api/recommendations/ff')
def index229837423764():
    print("len(map_productId_index)")
    print(len(map_productId_index))
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
#     model = getReqs_ff()
    result = getRec_ff(productIdsNew, model, 99)
    print(result)
    return result

@app.route('/browses')
def index3():    
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    print("productIdsNew")
    print(productIdsNew)
    returner = "" 
    returner += "input: <br/>\n"
    count = 0
    for productId in productIdsNew:
        count += 1
        print(productId)
        if productId in map_productId_imgUrl:
            url = map_productId_imgUrl[productId]
            url = map_productId_imgUrl[productId]
            print(url)
            returner += str(count) + " <img src='"+ url + "' width='100' />"+ productId+" </br>"
        else:
            returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+productId+" <br/>\n"
    returner += "customer browses: <br/><br/>\n"

    for customerId, tuples in map_customerId_productIdDateTuplesList.items():
        list = [tuple[0] for tuple in map_customerId_productIdDateTuplesList[customerId]]
        skipThisCustomer = False
        for productId in productIdsNew:
            if productId not in list:
                skipThisCustomer = True
                break
        if skipThisCustomer:
            continue        
        tuples = map_customerId_productIdDateTuplesList[customerId]
        customerId = str(customerId)
        returner += "<br/><h1>customerId: " +customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " " + customerId + " "+"</h1><br/>"
        for tuple in tuples:
            productId = tuple[0]
            timestamp = tuple[1].strftime("%Y-%m-%d %H:%M:%S")
            if productId in map_productId_imgUrl:
                url = map_productId_imgUrl[productId]
                returner += "<img src='"+ url + "' width='100' />"+ timestamp + " &nbsp;&nbsp;&nbsp; " + productId+"<br/>\n"
            else:
                returner += "no image "+timestamp + "<br/>\n"
    return returner    



import requests
from urllib.parse import quote
import json

# urlPre = 'http://35.165.204.194:9200'
# url = urlPre + '/image/_search?pretty'
# json = '{"query":{"bool":{"must":[{"range":{"thumbWidth":{"gte":1}}}]}}}'
# 
# r = requests.post(url, json).json()
# count = 0
# 
# while r['hits']['total'] > 0 and count < 37000:
#     for doc in r['hits']['hits']:
#         count += 1

def getRecBronto(productIdsInput):
    productIdsNew = productIdsInput[:]
    if siteId=='35569':
        queryProductIds = [map_productId_parentProductId[id] for id in productIdsNew]
    else:
        queryProductIds = productIdsNew
    #http://robinson.brontolabs.local:8983/solr/films/select?q=vv:"Covers for Modular Outdoor Club Chairs" && vv:"Resort Club & Ottoman Set"
    qEquals = "vv:("
    for id in queryProductIds:
        qEquals += '"'+id+'" '
#     qEquals = qEquals[:-1]
    qEquals += ")"
    qEquals = quote(qEquals)
    url = 'http://robinson.brontolabs.local:8983/solr/films/select?rows=200&q=siteId:'+siteId+' AND ' + qEquals
    print(url)
    r = requests.get(url).json()
    print(json.dumps(r, sort_keys=True, indent=4))
    print(url)
    i = 0
    if r["response"]["numFound"]==0:
        return None

    recProductId = r["response"]["docs"][i]["productId"]
    
    if siteId=='35569':
        productIdToCheck = map_productId_parentProductId[recProductId]
    else:
        productIdToCheck = recProductId
        
    while productIdToCheck in queryProductIds and i < len(r["response"]["docs"]) - 1:
        i += 1
        recProductId = r["response"]["docs"][i]["productId"]  
        if siteId=='35569':
            print("recProductId")
            print(recProductId)
            if recProductId not in map_productId_parentProductId:
                continue
            productIdToCheck = map_productId_parentProductId[recProductId]
        else:
            productIdToCheck = recProductId  
    return recProductId

def getRecsBronto(productIds, n=10):
    generated = productIds[:]
    for i in range(0,n):
        rec = getRecBronto(generated)
        if rec == None:
            return generated
        generated.append(rec)
    return generated

@app.route('/recommendations/bronto')
def index4():  
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    print("productIdsNew")
    print(productIdsNew)
    returner = "" 
    returner += "input: <br/>\n"
    for productId in productIdsNew:
        print(productId)
        if productId in map_productId_imgUrl:
            url = map_productId_imgUrl[productId]
            url = map_productId_imgUrl[productId]
            print(url)
            returner += "<img src='"+ url + "' width='100' />"#+ productId+" </br>"
        else:
            returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"#+productId+" <br/>\n"
    returner += "<br/>bronto recs: <br/><br/>\n"
    recProductIds = getRecsBronto(productIdsNew, 13)
    
    for recProductId in recProductIds:
        if recProductId in map_productId_imgUrl:
            url = map_productId_imgUrl[recProductId]
            print(url)
            returner += "<img src='"+ url + "' width='100' />"+ recProductId+" </br>"
        else:
            returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+recProductId+" <br/>\n"
# 
#     returner += "<pre>" + recProductId + "</pre></br>"
#     r = json.dumps(r, sort_keys=True, indent=4)
#     returner += "<pre>" + str(r) + "</pre>"
    return returner

@app.route('/display')
def index5():
    print('/display')
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    print(productIdsNew)
    returner = str(productIdsNew) + "</br>"
    returner += "input</br>"
    for productId in productIdsNew:
        returner += "<img src='"+ (map_productId_imgUrl[productId] if productId in map_productId_imgUrl else 'not found') + "' width='100' /> "+productId+"</br>"
    return returner
    
import random
@app.route('/random')
def index6():
    print('/display')
    productIdRandom = list(map_productId_imgUrl.keys())[random.randint(0, len(map_productId_imgUrl) - 1)]
    return "<img src='"+ (map_productId_imgUrl[productIdRandom] if productIdRandom in map_productId_imgUrl else 'not found') + "' width='100' /> "+productIdRandom+"</br>"


if __name__ == '__main__':
    app.run(debug=True)    


#sad comparisons
'''
http://127.0.0.1:5000/recommendations/bronto?productIds=OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L
http://127.0.0.1:5000/recommendations/lstm?diversity=1.0&productIds=OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L

good comparisons? using word len better than 512 for nn layers

http://127.0.0.1:5000/recommendations/lstm?diversity=1.0&productIds=OP-PECLB2M-BAR-YLW-K,OP-PECLB5M-BAR-YLW-K
http://127.0.0.1:5000/recommendations/bronto?productIds=OP-PECLB2M-BAR-YLW-K,OP-PECLB5M-BAR-YLW-K
http://127.0.0.1:5000/recommendations/lstmparentids?diversity=1.0&productIds=OP-PECLB2M-BAR-YLW-K,OP-PECLB5M-BAR-YLW-K

ubrellas:
http://127.0.0.1:5000/recommendations/bronto?productIds=OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L

white chairs:
http://127.0.0.1:5000/recommendations/bronto?productIds=ZM100334
http://127.0.0.1:5000/recommendations/lstmparentids?diversity=1.0&productIds=ZM100334

weird ... why switching.  like searching in wrong order http://127.0.0.1:5000/recommendations/bronto?productIds=OP-PESS9-CNS-NVY-K,OP-PEOSS6MTD-CNS-MXM-K,OP-PESS8-CNS-WIS-K,OP-PESS3-LUM-K,OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L
'''


