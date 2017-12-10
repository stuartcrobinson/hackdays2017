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
import tensorflow as tf
import numpy as np
import os
import os.path as osp
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model
from keras import backend as K

'''
#start elasticsearch:
sudo -u bronto sh /usr/local/bronto/commerce-es/bin/elasticsearch -d
#start solr
~/solr/solr-7.1.0/bin/solr start -c -p 8983 -s example/cloud/node1/solr
~/solr/solr-7.1.0/bin/solr start -c -p 7574 -s example/cloud/node2/solr -z localhost:9983
'''


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

map_customerId_productIdDateTuplesList = numpy.load('~/ml/dataMedium/map_customerId_productIdDateTuplesList_'+siteId+'.npy').item()
# map_customerId_productIdDateTuplesList = numpy.load('~/ml/dataMedium/map_newCustomerId_productIdDateTuplesList_'+siteId+'.npy').item()


map_productId_index = numpy.load('~/ml/dataMedium/map_productId_index_'+siteId+'.npy').item()
map_index_productId = dict((v,k) for (k,v) in map_productId_index.items())
map_productId_imgUrl = numpy.load('~/ml/dataMedium/map_productId_imgUrl_'+siteId+'.npy').item()
map_productId_parentProductId = numpy.load('~/ml/dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()
map_parentProductId_productId = dict((parentProductId, productId) for productId, parentProductId in map_productId_parentProductId.items())

if siteId=='35569':
    keras_model_file = '~/ml/model_dir/ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2.h5'
if siteId=='38178':
    keras_model_file = ''

tf_model_path = convert_to_pb(keras_model_file,'~/ml/model_dir/','~/ml/model_dir2/')
tf_model,tf_input,tf_output = load_graph(tf_model_path)

def getRec_ff(productIdsNew):
    Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
    for productId in productIdsNew:
        if productId not in map_productId_index:
            return "failed: productId not found in browse data: " + productId
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    
    print("productIdsNew:  gfdgfdkg")
    print(productIdsNew)
    prediction = predictFromTfModel(tf_model,tf_input,tf_output, Xnew)[0]
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    
    next_productIndex = numpy.argmax(prediction)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs_ff(productIdsInput, n=2):
    print("productIdsInput:  FYISDFSDF")
    print(productIdsInput)
    productIds = productIdsInput[:]
    for i in range(0, n):
        rec = getRec_ff(productIds)
        if "failed" in rec:
            return rec
        productIds.append(getRec_ff(productIds))
    return productIds

    
#!flask/bin/python
from flask import Flask

app = Flask(__name__)

@app.route('/api/productsdetails')
def hxrgs():
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")

    #TODO have to decide on elasticsearch vs solr for this first!!!!!
    # 
    
    
    return json.dumps({'productIds': result})


@app.route('/api/recommendations/ff')
def fwefs():
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
    numProductsToRecommend = request.args.get('n', default = 1, type = int)
    recProductIds = getRecs_ff(productIdsInput, n=numProductsToRecommend)
    return json.dumps({'productIds': result})


@app.route('/api/recommendations/bronto/solr')
def rhthr():
    productIdsInput = request.args.get('productIds', default = 'default', type = str).split(",")
    numProductsToRecommend = request.args.get('n', default = 1, type = int)
    search_engine = request.args.get('search_engine', default = 'solr', type = str)

    recProductIds = getRecsBronto(productIdsInput, n=numProductsToRecommend, search_engine=search_engine)
    return json.dumps({'productIds': recProductIds})
    


@app.route('/recommendations/ff')
def index2():
    print("len(map_productId_index)")
    print(len(map_productId_index))
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    result = getRecs_ff(productIdsNew, n=1)
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

def getRecommendedProductFromSearchEngine(productIds, search_engine="solr"):
    if search_engine=="solr":
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
        return recProductId
    
    if search_engine=="elasticsearch":
        return "not supported yet"

def getRecBronto(productIdsInput, search_engine="solr"):
    productIdsNew = productIdsInput[:]
    if siteId=='35569':
        queryProductIds = [map_productId_parentProductId[id] for id in productIdsNew]
    else:
        queryProductIds = productIdsNew
        
        
    
    
    recProductId = getRecommendedProductFromSearchEngine(productIds=queryProductIds, search_engine)
    
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

def getRecsBronto(productIds, n=10, search_engine="solr"):
    generated = productIds[:]
    for i in range(0,n):
        rec = getRecBronto(generated, search_engine=search_engine)
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


