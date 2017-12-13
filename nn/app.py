#!flask/bin/python
from flask import Flask
from flask import request
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
import datetime
import json
# import tensorflow as tf
import numpy as np
import os
import os.path as osp
# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io

# from keras.layers import Activation

# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint



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


def printProductIds(list):
    # http://127.0.0.1:5000/browses?productIds=OP-MKT10-PORIII-ET
    print("http://127.0.0.1:5000/browses?productIds=" + str(list).replace("'", "").replace(" ", "").replace("[","").replace("]",""))
    print()


if siteId=='38178':
#     weights = root + 'weights/ffWtfWeightsUsing1s_v2_hybridFalse_minTagCutoff0_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch2'
    weights = root + 'weights/ffWtfWeightsUsing1s_v2_hybridFalse_minTagCutoff0_usingParents_False_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch2'
if siteId=='35569':
    weights = root + 'weights/ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2'

map_customerId_productIdDateTuplesList = numpy.load(root + 'dataMedium/map_customerId_productIdDateTuplesList_usingParents_False_'+siteId+'.npy').item()
# map_customerId_productIdDateTuplesList = numpy.load('~/ml/dataMedium/map_newCustomerId_productIdDateTuplesList_'+siteId+'.npy').item()

map_productId_index = numpy.load(root + 'dataMedium/map_productId_index_usingParents_False_'+siteId+'.npy').item()
map_index_productId = dict((v,k) for (k,v) in map_productId_index.items())
map_productId_imgUrl = numpy.load(root + 'dataMedium/map_productId_imgUrl_'+siteId+'.npy').item()

map_productId_title = numpy.load(root + 'dataMedium/map_productId_title_'+siteId+'.npy').item()

map_productId_parentProductId = numpy.load(root + 'dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()
map_parentProductId_productId = dict((parentProductId, productId) for productId, parentProductId in map_productId_parentProductId.items())


'''
need these files:
~/ml/dataMedium/map_customerId_productIdDateTuplesList_usingParents_Fals*
~/ml/dataMedium/map_productId_index_usingParents_Fals*
~/ml/dataMedium/map_productId_imgUrl*
~/ml/dataMedium/map_productId_title_*
~/ml/dataMedium/map_productId_parentProductId_*
'''



idsToAddParentUrlsFor = []

for id in map_productId_imgUrl:
    if id not in map_productId_parentProductId:
        continue
    parentId = map_productId_parentProductId[id]
    if parentId not in map_productId_imgUrl:
        idsToAddParentUrlsFor.append(id)

for id in idsToAddParentUrlsFor:
    map_productId_imgUrl[map_productId_parentProductId[id]] = map_productId_imgUrl[id]
    
    


print("baseline_model start ...")
model = Sequential()
model.add(Dense(2*len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
model.add(Dense(len(map_productId_index), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.load_weights(weights)
print("baseline_model end____")

#### from ffNn.py 
def getRec_ff2(productIdsNew):
    Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    prediction = model.predict(Xnew, verbose=0)[0]
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    next_productIndex = np.argmax(prediction)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs_ff2(productIds, n):
    for i in range(0, n):
        productIds.append(getRec_ff2(productIds))
    return productIds
#### end from ffNn.py




# only submit products known to be in browse data
def getRec_ff(model, productIdsNew):
    Xnew = numpy.zeros(shape=(1, len(map_productId_index)))
    for productId in productIdsNew:
        if productId not in map_productId_index:
            return "failed: productId not found in browse data: " + productId
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    
    print("productIdsNew:  gfdgfdkg")
    print(productIdsNew)
    
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
    productIds = productIdsInput[:]
    print(' in getRecs_ff')
    print('input productIdsInput')
    print(productIdsInput)
    
    productIds = [x for x in productIds if x in map_productId_index]
    print(' but using:')
    print(productIds)
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


def getProductsDetailsOldStupidWithSearch(productIds):
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



def getProductsDetails(productIds):
    print('in getProductsDetails')
    productDocs = []
    for productId in productIds:
        doc = {}
        doc['productId'] = productId
        doc['productImageUrl'] = map_productId_imgUrl[productId] if productId in map_productId_imgUrl else 'no url'
        doc['productTitle'] = map_productId_title[productId] if productId in map_productId_title else 'no title'
        if productId in map_productId_imgUrl:
            productDocs.append(doc)
    return json.dumps(productDocs, indent=4)  # sort_keys=True,
        

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
    print("url")
    print(url)
    r = requests.get(url)
    data = r.json()
    docs = data['response']['docs']
#     print(data)
    outputDocs = []
    count = 1
    seenParentProductIds = set()
    for doc in docs:
        productId = doc['productId']
        productId = map_productId_parentProductId[productId]
        if productId in seenParentProductIds:
            continue
        seenParentProductIds.add(productId)
        if count > n:
            break
        if 'productImageUrl' in doc and productId in map_productId_index.keys():
            count += 1
            outputDocs.append({'productId': productId, 'productTitle': doc['productTitle'], 'productImageUrl': doc['productImageUrl'] if 'productImageUrl' in doc else ""})
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
    method = request.args.get('method', default = 1, type = int)
    
    productIdsInput = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsInput]

    
    print('/pythonapi/recommendations/ff')
    print('input products:')
    print(productIdsInput)
    
    if method == 1:
        recProductIds = getRecs_ff(productIdsInput, n=numProductsToRecommend)
    if method == 2:
        recProductIds = getRecs_ff2(productIdsInput, n=numProductsToRecommend)
    
    print('result: ')
    print(recProductIds)
    printProductIds(recProductIds)
    
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
#     [x+1 if x >= 45 else x+5 for x in l]
    productIdsInput = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsInput]
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
    print('/recommendations/ff')
    print("len(map_productId_index)")
    print(len(map_productId_index))
    
    productIdsNew = request.args.get('productIds', default = 'default', type = str).split(",")
    numProductsToRecommend = request.args.get('n', default = 1, type = int)

    method = request.args.get('method', default = 1, type = int)
    
    productIdsNew = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsNew]

    
    print('input products:')
    print(productIdsNew)
    
    if method == 1:
        result = getRecs_ff(productIdsNew, n=numProductsToRecommend)
    if method == 2:
        result = getRecs_ff2(productIdsNew, n=numProductsToRecommend)
    
    print('result: ')
    print(result)
    printProductIds(result)
    
    print("finished getting recs  lkjlkjlkj")
    if "failed" in result:
        return result
        
    returner = "Hello, World! " + str(productIdsNew) + "\n"
    returner += "browsed: <br/>\n"
    
    count = 0
    for productId in productIdsNew:
        count += 1
        print("productId")
        print(productId)
        if productId in map_productId_imgUrl:
            url = map_productId_imgUrl[productId]
            print(url)
            returner += str(count) + " <img src='"+ url + "' width='100' />"+ productId+" </br>"
        else:
            returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+productId+" <br/>\n"
    
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
    print('getRecommendedProductFromSearchEngine')
    if search_engine=="solr":
        #http://robinson.brontolabs.local:8983/solr/films/select?q=vv:"Covers for Modular Outdoor Club Chairs" && vv:"Resort Club & Ottoman Set"
        qEquals = "vv_indicators:("
        for id in productIds:
#             qEquals += 'vv_indicators:"' + quote(id) + '"%20'
            qEquals += '"' + (id) + '" '
#         qEquals = quote(qEquals)
        qEquals += ")"
        # qEquals = ""
#         for id in productIds:
# #             qEquals += 'vv_indicators:"' + quote(id) + '"%20'
#             qEquals += 'vv_indicators:"' + quote(id) + '"%20'
# #         qEquals = quote(qEquals)
#         qEquals += ""
        url = 'http://52.90.195.143:8983/solr/products/select?rows='+str(len(productIds)+5)+'&q=%2BproductImageUrl:*%20'+qEquals
        print('url')
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
            
        while productIdToCheck in productIds and i < len(r["response"]["docs"]) - 1:
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
    for i in range(0, n):
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
    numProductsToRecommend = request.args.get('n', default = 2, type = int)

    productIdsNew = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsNew]


    print('/recommendations/bronto')
    print("productIdsNew")
    print(productIdsNew)
    
    result = getRecsBronto(productIdsNew, numProductsToRecommend) 
    
#     returner = "" 
#     returner += "input: <br/>\n"
#     for productId in productIdsNew:
#         print(productId)
#         if productId in map_productId_imgUrl:
#             url = map_productId_imgUrl[productId]
#             print('url')
#             print(url)
#             returner += "<img src='"+ url + "' width='100' />"#+ productId+" </br>"
#         else:
#             returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"#+productId+" <br/>\n"
#     returner += "<br/>bronto recs: <br/><br/>\n"
#     
#     printProductIds('recProductIds')
#     printProductIds(results)
#     
#     for recProductId in results:
#         if recProductId in map_productId_imgUrl:
#             url = map_productId_imgUrl[recProductId]
#             print(url)
#             returner += "<img src='"+ url + "' width='100' />"+ recProductId+" </br>"
#         else:
#             returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+recProductId+" <br/>\n"
# # 
# #     returner += "<pre>" + recProductId + "</pre></br>"
# #     r = json.dumps(r, sort_keys=True, indent=4)
# #     returner += "<pre>" + str(r) + "</pre>"
#     return returner
# 
# 
# #     
#             
    returner = "Hello, World! " + str(productIdsNew) + "\n"
    returner += "browsed: <br/>\n"
    
    count = 0
    for productId in productIdsNew:
        count += 1
        print("productId")
        print(productId)
        if productId in map_productId_imgUrl:
            url = map_productId_imgUrl[productId]
            print(url)
            returner += str(count) + " <img src='"+ url + "' width='100' />"+ productId+" </br>"
        else:
            returner += "no image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+productId+" <br/>\n"
    
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
    app.run(host='0.0.0.0', debug=False, port=5000)


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


