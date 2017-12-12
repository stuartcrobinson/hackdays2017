# 
# #don't use gpu
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from __future__ import print_function
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy import sparse
from collections import defaultdict
from collections import Counter
import datetime
import random
import glob
import csv
import sys  
import io
import html
import string
import numpy
import keras
import time
import os
import copy

import numpy
import pandas
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint

root = os.path.expanduser('~/ml/')

if not os.path.exists(root + 'dataMedium'): #for app.py - web
    os.makedirs(root + 'dataMedium')        #need blank next line for pasting

if not os.path.exists(root + 'model_dir'): #for app.py - web
    os.makedirs(root + 'model_dir')        #need blank next line for pasting

#/home/ec2-user/ml/model_dir/ffWtfWeightsUsing1s_v2_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_35569_epoch1.h5

# siteId = "35569"
siteId = "38178"
# 
map_productId_parentProductId = numpy.load(root + 'dataMedium/map_productId_parentProductId_'+siteId+'.npy').item()

def printProductIds(list):
    # http://127.0.0.1:5000/browses?productIds=OP-MKT10-PORIII-ET
    print("http://127.0.0.1:5000/browses?productIds=" + str(list).replace("'", "").replace(" ", "").replace("[","").replace("]",""))
    print()


#dataGenCustomNNshort is for trying to get the NN to work at all.  original model w/ all products and all users and all item features didn't work. paring it down

hybrid = False
titlesOnly = True
minimumTagCount = 0
usingParents = False


#####################################################################################################################
# 1.  get item features.  build tagIndex->tag map
# 
# 
# 
# stemmer = PorterStemmer()
# 
# stopwords = ['qualiti', 'live', 'use', 'exist', 'allow', 'ad', 'start', 'make', 'way', 'rst', 'll', 're', 'd', 've', 's', 't', 'br', 'li', 'nbsp', 'p', 'span', 'div', 'ul', 'ol', 'includes','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']
# 
# def get_words(doc):
#     replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))     # replace punctuation with space
#     doc = doc.translate(replace_punctuation)
#     tokens = doc.split()                                                                            # split into tokens by white space
#     tokens = [word for word in tokens if (word.isalpha() and word not in stopwords)]       # remove remaining tokens that are not alphabetic
#     tokens = [stemmer.stem(word.lower()) for word in tokens]
#     tokens = [word for word in tokens if word not in stopwords]
#     return tokens
# 
# map_productId_descriptionTokens = {}
# path = root + "fordocker/products_production_siteId_=" + siteId + "/*.csv"
# nonascii = bytearray(range(0x80, 0x100))
# for fname in glob.glob(path):
#     print(fname)
#     with open(fname, mode='rb') as infile:
#         next(infile)
#         for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
#             line = line.translate(None, nonascii)
#             line = str(line,'utf-8')
#             for row in csv.reader([line], doublequote=True, quoting=csv.QUOTE_ALL, escapechar='\\'):
#                 productId = row[0]
#                 title = row[2]
#                 description = row[3] if siteId == "35569" else ""
#                 k,v  = productId, get_words(title) + get_words(description)[:13] 
#                 map_productId_descriptionTokens[k] = v
# 
# map_productId_descriptionTokens_original = copy.deepcopy(map_productId_descriptionTokens)
# 
# # now remove all tags that only exist once in the collection
# allTagsList = []
# for key in map_productId_descriptionTokens:
#     allTagsList.extend(map_productId_descriptionTokens[key])
# 
# map_tag_count = Counter(allTagsList)
# 
# minimumTagCount = 5
# 
# minimalUseTags = [tag for (tag, count) in map_tag_count.items() if count < minimumTagCount]
# 
# for key in map_productId_descriptionTokens:
#     tokens = map_productId_descriptionTokens[key]
#     shrunkTokens = [tag for tag in tokens if tag not in minimalUseTags]
#     map_productId_descriptionTokens[key] = shrunkTokens
# 
# # now rebuild tag stuff w/ shortened tags lists (after removing singles)
# 
# ##------------------------------------------------------------------------------------------------------------copy stop here on local
# 
# def getUniqueTags(map_productId_descriptionTokens):
#     shared = {}
#     allTagsSet = {}
#     # get starter description
#     for key in map_productId_descriptionTokens:
#         shared = set(map_productId_descriptionTokens[key])
#         allTagsSet = set(map_productId_descriptionTokens[key])
#         break
#     for key in map_productId_descriptionTokens:
#         shared = shared.intersection(map_productId_descriptionTokens[key])
#         allTagsSet = allTagsSet.union(map_productId_descriptionTokens[key])
#     return list(allTagsSet)
# 
# uniqueTagsList = getUniqueTags(map_productId_descriptionTokens)
# 
# 
# map_index_tag = dict(enumerate(uniqueTagsList))
# map_tag_index = {x:i for i,x in enumerate(uniqueTagsList)}
# 
# 
# 
#### end item features getting -- now need to incorporate features in the neural network.  need a node per token/tag
######################################################################################################################################################################
######################################################################################################################################################################
# 2. get browse lists

map_customerId_productIdDateTuplesList_original = defaultdict(list)

# try:
#     map_customerId_productIdDateTuplesList_original = numpy.load(root + 'dataMedium/map_customerId_productIdDateTuplesList_original_'+siteId+'.npy').item()
#     print("read map_customerId_productIdDateTuplesList_original from file")
# except IOError:
if True:
    # path = os.path.expanduser("~/ml/fordocker/browse_production_siteId_=35569/part-r-00016-55b1cd2d-d2c7-43dc-ac2a-da953f82d47b.csv")
    path = os.path.expanduser(root + "fordocker/browse_production_siteId_="+siteId+"/*.csv")
    nonascii = bytearray(range(0x80, 0x100))
    for fname in glob.glob(path):
        print(fname)
        with open(fname, mode='rb') as infile:
            next(infile)
            for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
                line = line.translate(None, nonascii)
                line = str(line,'utf-8')
                for row in csv.reader([line]):
                    customer_id = row[0]
                    product_id = row[2]
                    if usingParents and product_id in map_productId_parentProductId:
                        product_id = map_productId_parentProductId[product_id]
                    browse_timestamp = row[3]
                    format_date = datetime.datetime.strptime(browse_timestamp, '%Y-%m-%dT%H:%M:%S.%f%z')
                    map_customerId_productIdDateTuplesList_original[customer_id].append((product_id, format_date))
    map_customerId_productIdDateTuplesList_original = dict((id, list) for id,list in map_customerId_productIdDateTuplesList_original.items() if len(list) > 1)
    print("saving map_customerId_productIdDateTuplesList_original to file")
    numpy.save(root + 'dataMedium/map_customerId_productIdDateTuplesList_original_'+siteId+'.npy', map_customerId_productIdDateTuplesList_original)

print("coping map_customerId_productIdDateTuplesList from original")
map_customerId_productIdDateTuplesList = copy.deepcopy(map_customerId_productIdDateTuplesList_original)

print("sorting browse lists in map_customerId_productIdDateTuplesList ....")
for customerId, tuplesList in map_customerId_productIdDateTuplesList.items():
    map_customerId_productIdDateTuplesList[customerId] = sorted(tuplesList, key=lambda tup: tup[1])

secondsPause = 1800

newCustomerId = 0
map_newCustomerId_productIdDateTuplesList = defaultdict(list)
print("chopping browse lists in map_customerId_productIdDateTuplesList at pauses of " + str(secondsPause) + " seconds...")
for sortedTuples in map_customerId_productIdDateTuplesList.values():
    prevDatetime = datetime.datetime(1984, 2, 1, 15, 16, 17, 345, tzinfo=datetime.timezone.utc)
    for tuple in sortedTuples:
        currDatetime = tuple[1]
        if (currDatetime - prevDatetime).seconds > secondsPause:
            newCustomerId += 1
        map_newCustomerId_productIdDateTuplesList[newCustomerId].append(tuple)
        prevDatetime = currDatetime

# ################################################################################################################################################
# ################################################################################################################################################

print("removing single-item lists")
map_newCustomerId_productIdDateTuplesList = dict((id, list) for id,list in map_newCustomerId_productIdDateTuplesList.items() if len(list) > 1)

allProducts0 = set(tuple[0] for tuples in map_customerId_productIdDateTuplesList_original.values() for tuple in tuples)
allProducts = set([tuple[0] for tuples in map_newCustomerId_productIdDateTuplesList.values() for tuple in tuples])       #was list of set        

print("adding any missing products from original full browse lists (not pause-chopped)")
missingProductIds = allProducts0.difference(allProducts)
newCustomerIdBeforeAddingMissing = newCustomerId
if len(missingProductIds) > 0:
    for userId, tuples in map_customerId_productIdDateTuplesList_original.items():
        if len(tuples) > 0:
            productsList =  [tuple[0] for tuple in map_customerId_productIdDateTuplesList[userId]]
            for missingProduct in missingProductIds:
                if missingProduct in productsList:
                    newCustomerId += 1
                    map_newCustomerId_productIdDateTuplesList[newCustomerId] = tuples
                    continue
    allProducts = set([tuple[0] for sublist in map_newCustomerId_productIdDateTuplesList.values() for tuple in sublist])   #was list of set        

#------------------------------------------------------------------------------------------------------------------------------------------------------

allUsers = set(map_newCustomerId_productIdDateTuplesList.keys()) #was list of set        

map_customerId_productIdDateTuplesList = map_newCustomerId_productIdDateTuplesList

# # nonchop stuff
# print("removing single-item lists")
# map_customerId_productIdDateTuplesList = map_customerId_productIdDateTuplesList_original
# allProducts = set([tuple[0] for tuples in map_customerId_productIdDateTuplesList_original.values() for tuple in tuples])
# allUsers = set(map_customerId_productIdDateTuplesList.keys())


map_userId_index = {}
map_index_userId = {}

for i, userId in enumerate(allUsers):
    map_userId_index[userId] = i
    map_index_userId[i] = userId

# stupid to not use comprehension, but mandatory cos doens't break console copy/paste.  ???
# map_productId_index = {}
# map_index_productId = {}
# 
# for i, productId in enumerate(allProducts):
#     map_productId_index[productId] = i
#     map_index_productId[i] = productId

# this stuff is cool but it breaks copy/paste into terminal :(
map_productId_index = dict((id, i) for (i, id) in enumerate(allProducts))
map_index_productId = dict((i, id) for (i, id) in enumerate(allProducts))


map_userID_map_productIndex_count = {}
for userIndex, userId in enumerate(allUsers):
    productsList = [tuple[0] for tuple in map_customerId_productIdDateTuplesList[userId]]
    if len(productsList) < 2:
        continue
    map_productIndex_count = Counter([map_productId_index[id] for id in productsList])
    map_userID_map_productIndex_count[userId] = map_productIndex_count


# print("writing map_productId_index and map_customerId_productIdDateTuplesList")
numpy.save(root + 'dataMedium/map_productId_index_usingParents_'+str(usingParents)+'_'+siteId+'.npy', map_productId_index)
numpy.save(root + 'dataMedium/map_customerId_productIdDateTuplesList_usingParents_'+str(usingParents)+'_'+siteId+'.npy', map_customerId_productIdDateTuplesList)


#####################################################################################################################
#####################################################################################################################
# 3.  create matrices
    
#how long should x be?  each user should get n rows, where n is the number of products browsed.
# number of rows in x is the same as the number of training samples.  one per product in browse group (either entire browses of a user, or browses split up by hour pauses)  remember 
#     that the lists in map_customerId_productIdList contain repeated products!!!!!!!!!
#xRows should be the total number of unique item browses per user.  so if bob looked at 2 cats and a dog, and matt looked at a cat, dog, and 3 fish, xRows = 5

numProductInteractionTrainingSamples = (sum(len(map_productIndex_count) for map_productIndex_count in map_userID_map_productIndex_count.values()))

# browsedProductsWithFeatures = set(allProducts).intersection(map_productId_descriptionTokens.keys())
# numBrowsedProductsWithFeatures = len(browsedProductsWithFeatures)

if hybrid:
    XNumRows = numProductInteractionTrainingSamples + numBrowsedProductsWithFeatures
else:
    XNumRows = numProductInteractionTrainingSamples

numInputNodes = len(allProducts) + len(map_tag_index) if hybrid else len(allProducts) 
numOutputNodes = len(allProducts)

X = numpy.zeros(shape=(XNumRows, numInputNodes), dtype=bool)    #YESSSS MAKING THIS BOOL TYPE REALLY HELPED!  it's fniding the CUDA device now
y = numpy.zeros(shape=(XNumRows, numOutputNodes), dtype=bool)

rowIndex = -1

for map_productIndex_count in map_userID_map_productIndex_count.values():
    yRow = numpy.zeros(numOutputNodes)
    for productIndex, count in map_productIndex_count.items():
        yRow[productIndex] = 1 #count
    for productIndex, count in map_productIndex_count.items():
        rowIndex += 1
        X[rowIndex][productIndex] = 1 #count
        y[rowIndex] = yRow[:]
        if hybrid:
            productId = map_index_productId[productIndex] 
            if not productId in map_productId_descriptionTokens:
                continue
            descriptionTags = map_productId_descriptionTokens[productId]
            for tag in descriptionTags:
                tagIndex = map_tag_index[tag]
                nodeIndex = len(allProducts) + tagIndex
                X[rowIndex][nodeIndex] = 1 #count

#now add a training sample per product with only description features mapping to the product
if hybrid:
    for productId in browsedProductsWithFeatures:
        rowIndex += 1
        descriptionTags = map_productId_descriptionTokens[productId]
        for tag in descriptionTags:
            tagIndex = map_tag_index[tag]
            nodeIndex = len(allProducts) + tagIndex
            X[rowIndex][nodeIndex] = 1 #count

        



seed = 7                  # fix random seed for reproducibility
numpy.random.seed(seed)

X2 = X
y2 = y


import socket
# from sklearn import model_selection
# from sklearn.model_selection import train_test_split


# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=7)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)



# X_sparse_csr = sparse.csr_matrix(X)
# y_sparse_csr = sparse.csr_matrix(y)

def baseline_model():
    model = Sequential()
#     model.add(Dropout(0.5, input_shape=(len(map_productId_index),)))
    model.add(Dense(2*numInputNodes, input_dim=numInputNodes, activation='relu'))
#     model.add(Dropout(0.5))
    model.add(Dense(len(map_productId_index), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model = baseline_model()
totalIterations = 0

# define the checkpoint
# filepath="weightsWithItemFeatures-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
## fit the model# 
# modelWithItemFeatures = baseline_model()
# modelWithItemFeatures.fit(X2, y2, epochs=100, batch_size=100, callbacks=callbacks_list)



# model.load_weights(root + 'ffWtfWeightsUsing1s_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_USDR00253_35569_epoch2 allProducts
np = numpy

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

def getRecs(productIdsInput, n):
    productIdsInput = [map_productId_parentProductId[x] if x in map_productId_parentProductId else x for x in productIdsInput]
    productIds = productIdsInput[:]
    for i in range(0, n):
        productIds.append(getRec(productIds))
    return productIds


# model = load_model(root + 'model_dir/ffWtfWeightsUsing1s_v2_hybridTrue_minTagCutoff5_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch3.h5')
# printProductIds(getRecs(['ZM100334'], 30))



for iteration in range(1, 5):
    totalIterations += 1
    print("iteration: " + str(totalIterations))
    model.fit(X, y, epochs=1, batch_size=2000 ) #validation_data=(X_test,y_test)#fit shuffles by default according to https://stackoverflow.com/questions/42709051/what-does-the-acc-means-in-the-keras-model-fit-output-the-accuracy-of-the-final#comment72552203_42709115
    name = 'ffWtfWeightsUsing1s_v2_hybrid'+str(hybrid)+'_minTagCutoff'+str(minimumTagCount)+'_usingParents_'+str(usingParents)+'_NoDrpOut_1HdnLyr_DoubleLen1stLyr_'+str(secondsPause)+"sPauseChops_" + str(socket.gethostname()) + "_" + siteId + "_epoch" + str(totalIterations)
    model.save_weights(name, overwrite=True)
#     model.save_weights(name, overwrite=True)
#     model.save(root + 'model_dir/'+ name + '.h5')  # creates a HDF5 file 'my_model.h5'    https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    
    if siteId=='35569':
        printProductIds(getRecs(['OP-MKT10-PORIII-ET-K','OP-MKT12-PORIII-K','OP-MKT6510AT-HBG'], 30))
#         printProductIds(getRecs(['OP-MKT10-PORIII-ET-K','OP-MKT12-PORIII-K','OP-MKT6510AT-HBG'], 30))
#         printProductIds(getRecs(['OP-MKT10-PORIII-ET-K','OP-MKT12-PORIII-K','OP-MKT6510AT-HBG'], 30))
    if siteId=='38178':
        printProductIds(getRecs(['ZM100334'], 30)) #110374
        printProductIds(getRecs(['110374'], 30)) #110374
#         printProductIds(getRecs(['ZM100334'], 30))
#         printProductIds(getRecs(['ZM100334'], 30))
    print('writing')



# ffWtfWeightsUsing1s_hybrid_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_35569_epoch2.h5

# nohup python3 nn.py > logNn.log 2>&1 &

# model.fit(X2, y2, epochs=200, batch_size=10, callbacks=callbacks_list)  #starting at 4.44 - looking better after logout, reboot
# model.fit(X2, y2, epochs=10, batch_size=10000)

#####################################################################################################################################
#####################################################################################################################################
## getting recommendations
#
#

printProductIds(getRecs(['ZM100334'], 10))
from keras.models import load_model


model = load_model(root + 'model_dir/ffWtfWeightsUsing1s_v2_hybridTrue_minTagCutoff5_NoDrpOut_1HdnLyr_DoubleLen1stLyr_1800sPauseChops_ip-172-31-25-45_38178_epoch1.h5')


printProductIds(getRecs(['OP-MKT10-PORIII-ET-K','OP-MKT12-PORIII-K','OP-MKT6510AT-HBG'], 100)) #covers
printProductIds(getRecs(['OP-FCCT2646','OP-FCSS7-VST-K'], 100)) #covers

# OP-MKT10-PORIII-ET-K,OP-MKT12-PORIII-K,OP-MKT6510AT-HBG #umbrellas

printProductIds(getRecs(['OP-MKT10-PORIII-ET-K','OP-MKT10-PORII-K','SP-MKT10-L'], 10))

OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L
    

productIdsNew = ['IP-ACC16GRK-GRY'] # 'OP-PEBS2-CNS-SPA-K','OP-PEBST5-BLS-K','IP-PEBS2-PORIII-GRY'] #stools
#map_productId_title

getRecs(['IP-ACC16GRK-GRY'], 100)

getRecs(['OP-FCCT2646','OP-FCSS7-VST-K','OP-PESS6MFT-PORIII-LGB-K','IP-PESS4-MAN-K','OP-FC4PC-PORIII'])

# ZM100334 - highfahion
# 
# ubrellas:
# http://127.0.0.1:5000/recommendations/bronto?productIds=OP-MKT10-PORIII-ET-K,OP-MKT10-PORII-K,SP-MKT10-L



#     map_productIndex_count = defaultdict(int)
#     for productId in productsList:
#         map_productIndex_count[map_productId_index[productId]] += 1
#     productIndicesList = [map_productId_index[productId] for productId in productsList]
    # 
#     model.add(Dropout(0.5))                                                      # model.add(Dense(int(len(X2[0])))) #makes loss very slightly less.  by 0.01 at epoch 11
#     model.add(Dense(len(allProducts), activation='softmax'))