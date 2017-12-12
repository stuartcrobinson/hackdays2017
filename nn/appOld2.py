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
import json
# import tensorflow as tf
import numpy as np
import os
import os.path as osp
# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K



# siteId ='35569'# '38178' #'35569'
siteId ='38178'

'''
#start elasticsearch:
sudo -u bronto sh /usr/local/bronto/commerce-es/bin/elasticsearch -d
#start solr
~/solr/solr-7.1.0/bin/solr start -c -p 8983 -s example/cloud/node1/solr
~/solr/solr-7.1.0/bin/solr start -c -p 7574 -s example/cloud/node2/solr -z localhost:9983
'''
root = os.path.expanduser('~/ml/')


if siteId=='38178':
    weights = root + 'weights/ffWtfWeightsUsing1s_v2_hybridFalse_minTagCutoff0_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch2'
if siteId=='35569':
    weights = root + 'weights/ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2'

map_customerId_productIdDateTuplesList = numpy.load(root + 'dataMedium/map_customerId_productIdDateTuplesList_'+siteId+'.npy').item()
# map_customerId_productIdDateTuplesList = numpy.load('~/ml/dataMedium/map_newCustomerId_productIdDateTuplesList_'+siteId+'.npy').item()

map_productId_index = numpy.load(root + 'dataMedium/map_productId_index_'+siteId+'.npy').item()
map_index_productId = dict((v,k) for (k,v) in map_productId_index.items())
map_productId_imgUrl = numpy.load(root + 'dataMedium/map_productId_imgUrl_'+siteId+'.npy').item()
map_productId_parentProductId = numpy.load(root + 'dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()
map_parentProductId_productId = dict((parentProductId, productId) for productId, parentProductId in map_productId_parentProductId.items())


print("baseline_model start ...")
model = Sequential()
model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
model.add(Dense(len(map_productId_index), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("baseline_model end____")
# model._make_predict_function() 
print("_make_predict_function end____")

# 
# 
# def baseline_model():
#     print("baseline_model start ...")
#     model = Sequential()
#     model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
#     model.add(Dense(len(map_productId_index), activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     print("baseline_model end____")
#     return model
# 
# 
# 
# # Create function to convert saved keras model to tensorflow graph
# def convert_to_pb(h5_model_file, input_fld='',output_fld=''):
#    
#     h5_model_file = "myOutputFileName"
#     print("h5_model_file")
#     print(h5_model_file)
#     print("input_fld")
#     print(input_fld)
#     print("output_fld")
#     print(output_fld)
#    
#     # h5_model_file is a .h5 keras model file
#     output_node_names_of_input_network = ["pred0"] 
#     output_node_names_of_final_network = 'output_node'
#     
#     # change filename to a .pb tensorflow file
#     output_graph_name = h5_model_file[:-2]+'pb'                  
#     weight_file_path = osp.join(input_fld, h5_model_file)
#     
#     print("weight_file_path")
#     print(weight_file_path)
#     
# #     net_model = load_model(weight_file_path)
#     
#     net_model = baseline_model()
#     net_model.load_weights(weights)
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
#     print("frozen_graph_filename:")
#     print(frozen_graph_filename)
#     with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
#         print("f:")
#         print(f)
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
# def predictFromTfModel(tf_model, tf_input, tf_output, input_data, x, y):
#     # load tf graph
# #     tf_model,tf_input,tf_output = load_graph(model_path)
#     # 
# #     # Create tensors for model input and output
# #     x = tf_model.get_tensor_by_name(tf_input)
# #     y = tf_model.get_tensor_by_name(tf_output) 
#     
#     # Number of model outputs
#     num_outputs = y.shape.as_list()[0]
#     predictions = np.zeros((input_data.shape[0],num_outputs))
#     print('generating predictions START...')
#     with tf.Session(graph=tf_model) as sess:
#         for i in range(input_data.shape[0]):        
#             y_out = sess.run(y, feed_dict={x: input_data[i:i+1]})
#             predictions[i] = y_out
#     print('generating predictions FINISHED...')
#     return predictions
# 

# 
# 
# #OP-PEBS2-CHR-K
# 
# if siteId=='35569':
# #     keras_model_file = root + 'model_dir/ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2.h5'
#     keras_model_file = 'ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2.h5'
# if siteId=='38178':
#     keras_model_file = 'ffWtfWeightsUsing1s_hybridFalse_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch2.h5'
# 
# 
# keras_model_file = 'dummy var - using weights'
# # tf_input
# tf_model_path = convert_to_pb(keras_model_file, root +'model_dir/', root +'model_dir2/')
# tf_model,tf_input_name,tf_output_name = load_graph(tf_model_path)
# 
# 
#     # Create tensors for model input and output
# x = tf_model.get_tensor_by_name(tf_input_name)
# y = tf_model.get_tensor_by_name(tf_output_name) 
# 
# print('tf_model')
# print(tf_model)
# print('tf_input_name')
# print(tf_input_name)
# print('tf_output_name')
# print(tf_output_name)
# print('x is')
# print(x)
# print('y is')
# print(y)

#### from ffNn.py
def getRec(productIdsNew):
    Xnew = numpy.zeros(shape=(1, numInputNodes))
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
        if hybrid:
            if productId in map_productId_descriptionTokens:
                descriptionTags = map_productId_descriptionTokens[productId]
                for tag in descriptionTags:
                    tagIndex = map_tag_index[tag]
                    nodeIndex = len(allProducts) + tagIndex
                    Xnew[0][nodeIndex] = 1 #count        
    prediction = model.predict(Xnew, verbose=0)[0]
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    next_productIndex = np.argmax(prediction)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs(productIds, n):
    for i in range(0, n):
        productIds.append(getRec(productIds))
    return productIds
#### end from ffNn.py




def getRec_ff(model, productIdsNew):
    Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
    for productId in productIdsNew:
        if productId not in map_productId_index:
            return "failed: productId not found in browse data: " + productId
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    
    print("productIdsNew:  gfdgfdkg")
    print(productIdsNew)
#     prediction = predictFromTfModel(tf_model, tf_input_name, tf_output_name, Xnew, x, y)[0]
#     model = baseline_model()
#     print("baseline_model start ...")
#     model = Sequential()
#     model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
#     model.add(Dense(len(map_productId_index), activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     print("baseline_model end____")
# 
#     print("load weights start ....")
#     model.load_weights(weights)
#     print("load weights end_____")
    
    print("model.predict start ....")
    prediction = model.predict(Xnew, verbose=1)[0]
    print("model.predict end___")    
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    
    next_productIndex = numpy.argmax(prediction)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs_ff(productIdsInput, n=2):    
    print("baseline_model start ...")
#     model = Sequential()
#     model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
#     model.add(Dense(len(map_productId_index), activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     print("baseline_model end____")
#     
#     print("load weights start ....")
#     model.load_weights(weights)
#     print("load weights end_____")
#     model = 'dummy'
    productIds = productIdsInput[:]
    
    for i in range(0, n + len(productIdsInput)):
        rec = getRec_ff(model, productIds)
        if "failed" in rec:
            print('ff rec was none!!!!!!!!!!!') ######################################################################################## TODO start here.  why rec is none??? should i redo models using only products with urls and in feed and in browse list and with vv_indicators???
            return rec
        productIds.append(rec)
    returner = [x for x in productIds if x not in productIdsInput]
    return returner

    
#!flask/bin/python
import flask
from flask import Flask
import urllib.request

import codecs


app = Flask(__name__)


def getProductsDetails(productIds):
#     testurl = "http://robinson.brontolabs.local:8983/solr/products/select?pretty=true&q=productId:ZM100577%20productId:091130%20productId:063336%20"
#     testurl2 = "http://robinson.brontolabs.local:8983/solr/products/select?q=productId:ZM100577 productId:091130 productId:063336"   
#     return json.dumps({'productIds': result})
    print('in getProductsDetails')
    url1 = "http://robinson.brontolabs.local:8983/solr/products/select?q="
    for productId in productIds:
#         if productId in map_productId_index.keys():
        url1 += "productId:\"" + productId.replace(" ", "%20") + "\"%20"
    print("url")
    print(url1)
    with urllib.request.urlopen(url1) as url:
        data = json.loads(url.read().decode())
        docs = data['response']['docs']
        print('data')
        print(data)
        outputDocs = []
        for doc in docs:
            outputDocs.append({'productId': doc['productId'], 'productTitle': doc['productTitle'], 'productImageUrl': doc['productImageUrl'] if 'productImageUrl' in doc else ""})
        return json.dumps(outputDocs, indent=4)  # sort_keys=True,

# //not used.  too slow. compared to going straight from elasticsearch to node
@app.route('/pythonapi/suggest')
def hx2gs():
    q = request.args.get('q', type = str)
    r = requests.post('http://robinson.brontolabs.local:9115/products/_search?pretty', json={'suggest': {'mysuggest': {'prefix': q, 'completion': {'field': 'titlesuggest'}}}})
    data = r.json()
    options = data['suggest']['mysuggest'][0]['options']
    titles = (option['text'] for option in options)
    output = json.dumps(list(titles))
    resp = flask.Response(output)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/pythonapi/search_catalog')
def hxr7s():    
# &rows=2
    print("/pythonapi/search_catalog") 
    q = request.args.get('q', type = str)#.replace('"', '\\"')
    n = request.args.get('n', default=15, type = int)
#     http://robinson.brontolabs.local:8983/solr/products/select?indent=on&q=%2BproductImageUrl:*%20productTitle:(white%20chair)
# n is multiplied here cos lots of products in database (from browse data) aren't in the map_productId_index.  why, idk. 
    url = 'http://robinson.brontolabs.local:8983/solr/products/select?rows='+str(n*5)+'&q=%2BproductImageUrl:*%20%2BproductTitle:('+q+')'
    print(url)
    print("url")
    r = requests.get(url)
    data = r.json()
    docs = data['response']['docs']
#     print(data)
    outputDocs = []
    count = 1
    for doc in docs:
        if count > n:
            break
        if 'productImageUrl' in doc and doc['productId'] in map_productId_index.keys():
            count += 1
            outputDocs.append({'productId': doc['productId'], 'productTitle': doc['productTitle'], 'productImageUrl': doc['productImageUrl'] if 'productImageUrl' in doc else ""})
    resp = flask.Response(str(json.dumps(outputDocs, indent=4)))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
#     return json.dumps(outputDocs, indent=4)  # sort_keys=True,
    
# 
# @app.route('/api/products_details')
# def hx9gs():
#     productIdsInput = request.args.get('productIds', default = 'ZM100577|091130|063336', type = str).split("|")
#     return getProductsDetails(productIdsInput)


@app.route('/pythonapi/recommendations/ff')
def fwefs():
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split("|")
    productIdsInput = list(filter(None, productIdsInput))
    numProductsToRecommend = request.args.get('n', default = 1, type = int)
    doReturnDetails = request.args.get('return_details', default = "true", type = str)
    
    print('/pythonapi/recommendations/ff')
    print('productIdsInput: ')
    print(productIdsInput)
    recProductIds = getRecs_ff(productIdsInput, n=numProductsToRecommend)
    
    print('recProductIds: ')
    print(recProductIds)
    
    if doReturnDetails == "true":
        output = getProductsDetails(recProductIds)
    else:
        output = json.dumps({'productIds': recProductIds})

    resp = flask.Response(output)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/pythonapi/recommendations/bronto')
def rhthr():
    print('/pythonapi/recommendations/bronto')
    print('request.args')
    print(request.args)
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split("|")
    productIdsInput = list(filter(None, productIdsInput))
    numProductsToRecommend = request.args.get('n', default = 2, type = int)
    doReturnDetails = request.args.get('return_details', default = "true", type = str)

    recProductIds = getRecsBronto(productIdsInput, n=numProductsToRecommend, search_engine="solr")
    print("len(recProductIds)")
    print(len(recProductIds))
#     return json.dumps({'productIds': recProductIds})
    
    if doReturnDetails == "true":
        output = getProductsDetails(recProductIds)
        print("len(output)")
        print(len(output))
    else:
        output = json.dumps({'productIds': recProductIds})
    
    resp = flask.Response(output)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['lalallala'] = 'hieeeee'
    return resp

@app.route('/recommendations/ff')
def index2():
    print("len(map_productId_index)")
    print(len(map_productId_index))
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    result = getRecs_ff(productIdsNew, n=13)
    print("finished getting recs  lkjlkjlkj")
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

#     print(map_productId_imgUrl['OP-PESS4-CNS-SLT-K'])
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

#element 0 is strongest rec.  only search for elements with image url
def getRecommendedProductFromSearchEngine(productIds, search_engine="solr"):
    print("productIds")
    print(productIds)
    if search_engine=="solr":
        #http://robinson.brontolabs.local:8983/solr/films/select?q=vv:"Covers for Modular Outdoor Club Chairs" && vv:"Resort Club & Ottoman Set"
        qEquals = ""
        for id in productIds:
            qEquals += 'vv_indicators:"' + quote(id) + '"%20'
#         qEquals = quote(qEquals)
        url = 'http://robinson.brontolabs.local:8983/solr/products/select?q=%2BproductImageUrl:*%20'+qEquals
        print(url)
        r = requests.get(url).json()
#         print(json.dumps(r, sort_keys=True, indent=4))
        print(url)
        i = 0
        if r["response"]["numFound"]==0:
            return None
        recProductId = r["response"]["docs"][i]["productId"]
        
        if siteId=='35569':
            productIdToCheck = map_productId_parentProductId[recProductId]
        else:
            productIdToCheck = recProductId
            
        while productIdToCheck in productIds and i < len(r["response"]["docs"]):
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
    
    if search_engine=="elasticsearch":
        return "not supported yet"

def getRecBronto(productIdsInput, search_engine="solr"):
    print("productIdsInput")
    print(productIdsInput)
    productIdsNew = productIdsInput[:]
    if siteId=='35569':
        queryProductIds = [map_productId_parentProductId[id] for id in productIdsNew]
    else:
        queryProductIds = productIdsNew
    
    recProductId = getRecommendedProductFromSearchEngine(productIds=queryProductIds, search_engine=search_engine)
        
    return recProductId #TODO start here

def getRecsBronto(productIds, n=3, search_engine="solr"):
    print("getRecsBronto: productIds")
    print(productIds)
    generated = productIds[:]
    for i in range(0,n + len(productIds)):
        rec = getRecBronto(generated, search_engine=search_engine)
        if rec == None:
            print('bronto rec was none!!!!!!!!!!!') ######################################################################################## TODO start here.  why rec is none??? should i redo models using only products with urls and in feed and in browse list and with vv_indicators???
            return generated
        generated.append(rec)
    returner = [x for x in generated if x not in productIds]
    return returner


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
    app.run(debug=False)    


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


