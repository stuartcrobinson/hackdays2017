from __future__ import print_function
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy import sparse
from collections import defaultdict
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
siteId = "35569"

'''
dataGenCustomNNshort is for trying to get the NN to work at all.  original model w/ all products and all users and all item features didn't work. paring it down
'''

#### TODO
## - write/read data to file (like i'm doing.  stop re-processing data per run.     need to run discrete file on aws cos bad connection.  
## write some mac thing to reset wireless connection if internet lost
## tensorflow taking long time ... compile on machine?  trying rn w/ very small hidden layer.  but still takes 2+ minutes to prepare data. ran for 30 minutes so far ... nothing :( even with tiny hidden layer D:  need embedding??????
## start w/ smaller data?  no time to get new site ... just shrink existing data?  
## wtf why saying elapsed time - 263 seconds D:
## will adding dropout speed it up?

#####################################################################################################################
# 1.  get item features.  build tagIndex->tag map
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
# map_productId_title = {}
# path = "fordocker/products_production_siteId_=" + siteId + "/*.csv"
# nonascii = bytearray(range(0x80, 0x100))
# for fname in glob.glob(path):
#     print(fname)
#     with open(fname, mode='rb') as infile:
#         next(infile)
#         for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
#             line = line.translate(None, nonascii)
#             line = str(line,'utf-8')
#             for row in csv.reader([line]):
#                 k,v  = row[0], get_words(row[2]) + get_words(row[3])[:13] 
#                 map_productId_descriptionTokens[k] = v
#                 map_productId_title[k] = row[2]
# shared = {}
# allTagsSet = {}
# # get starter description
# for key in map_productId_descriptionTokens:
#     shared = set(map_productId_descriptionTokens[key])
#     allTagsSet = set(map_productId_descriptionTokens[key])
#     break
# 
# for key in map_productId_descriptionTokens:
#     shared = shared.intersection(map_productId_descriptionTokens[key])
#     allTagsSet = allTagsSet.union(map_productId_descriptionTokens[key])
# 
# #done making item features!!!!!!   what next?  collect all tags into a single list/vector.  now, made maps!  seeeeee
# 
# allTagsSet = list(allTagsSet)
# 
# map_index_tag = dict(enumerate(allTagsSet))
# map_tag_index = {x:i for i,x in enumerate(allTagsSet)}
# 

#####################################################################################################################
#####################################################################################################################
# 2. get browse lists


allProducts = set()
map_customerId_productIdList = defaultdict(list)
map_customerId_productIdDateTuplesList = defaultdict(list)

# if True:
# path = "fordocker/browse_production_siteId_=35569/part-r-00016-55b1cd2d-d2c7-43dc-ac2a-da953f82d47b.csv"
path = "fordocker/browse_production_siteId_="+siteId+"/*.csv"
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
                browse_timestamp = row[3]
                allProducts.add(product_id)
#                 map_customerId_productIdList[customer_id].append(product_id)
#datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%f%z')
                format_date = datetime.datetime.strptime(browse_timestamp, '%Y-%m-%dT%H:%M:%S.%f%z')
                map_customerId_productIdDateTuplesList[customer_id].append((product_id, format_date))


# 
# # do this part and comment out next code until numpy.save to use original customers.  not split by time
# for customer_id in map_customerId_productIdDateTuplesList:
#     tuplesList = map_customerId_productIdDateTuplesList[customer_id]
#     sortedTuples = sorted(tuplesList, key=lambda tup: tup[1])
#     map_customerId_productIdDateTuplesList[customer_id] = sortedTuples
#     products = [i[0] for i in sortedTuples]
#     map_customerId_productIdList[customer_id] = products
 

#now that browse data is sorted by time - and listed with time in tuple - break into browse sessions with maximum pause of 1 hour.    
    
#make new map_customerId_productIdDateTuplesList with matching map_customerId_productIdList
#how to do this?  for each tuples list, check time difference between consecutive.  #python how parse timestamp and subtract times
#https://stackoverflow.com/questions/44691220/python-parsing-and-converting-string-into-timestamp
#maybe tuple should be timestamp????? how to know if should store as string or datetime.  nah don't need to store as time permamently 

#TODO start here - think about how to label new customer ids- append smoething to original?  or make new ids?
    #just make new ids.   newCustomerId

newCustomerId = 0

map_newCustomerId_productIdDateTuplesList = defaultdict(list)
map_newCustomerId_productIdList = defaultdict(list)

for customer_id in map_customerId_productIdDateTuplesList:
    tuplesList = map_customerId_productIdDateTuplesList[customer_id]
#     print(tuplesList)
    sortedTuples = sorted(tuplesList, key=lambda tup: tup[1])
    #
    prevDatetime = datetime.datetime(1984, 2, 1, 15, 16, 17, 345, tzinfo=datetime.timezone.utc)
    for tuple in sortedTuples:
        currDatetime = tuple[1]
        if (currDatetime - prevDatetime).seconds > 3600:
            newCustomerId += 1
        map_newCustomerId_productIdDateTuplesList[newCustomerId].append(tuple)

map_customerId_productIdDateTuplesList = map_newCustomerId_productIdDateTuplesList
map_customerId_productIdList = defaultdict(list)

for newCustomer_id in map_newCustomerId_productIdDateTuplesList:
    tuplesList = map_newCustomerId_productIdDateTuplesList[newCustomer_id]
    if len(tuplesList) < 2:
        continue
    map_customerId_productIdDateTuplesList[newCustomer_id] = tuplesList
    sortedTuples = sorted(tuplesList, key=lambda tup: tup[1])
    products = [i[0] for i in sortedTuples]
    map_customerId_productIdList[newCustomer_id] = products
 




numpy.save('dataMedium/map_customerId_productIdDateTuplesList_'+siteId+'.npy', map_customerId_productIdDateTuplesList)
numpy.save('dataMedium/map_customerId_productIdList_'+siteId+'.npy', map_customerId_productIdList)
    

#now                 

#ok now i have lists of products per user ... now what .... 

#get set of users and set of products.  for amounts.

allUsers = list(set(map_customerId_productIdList.keys()))
allProducts = list(allProducts)

map_userId_index = {}
map_index_userId = {}

for i, userId in enumerate(allUsers):
    map_userId_index[userId] = i
    map_index_userId[i] = userId

map_productId_index = {}
map_index_productId = {}

for i, productId in enumerate(allProducts):
    map_productId_index[productId] = i
    map_index_productId[i] = productId


# map_productIndex_descriptionTokens = {}
# 
# for productId in map_productId_descriptionTokens:
#     try:
#         productIndex = map_productId_index[productId]
#         print(productId, productIndex)
#         map_productIndex_descriptionTokens[productIndex] = map_productId_descriptionTokens[productId]
#     except:
#         pass
#     
# 

# # # testing
# OP-PECLB2M-SUN-K,OP-PELOVO-SUN-K
# prodIndex1 = map_productId_index['OP-PECLB2M-SUN-K']
# prodIndex2 = map_productId_index['OP-PELOVO-SUN-K']
# 
# myset = set()
# 
# for r in range(0, XNumRows):
#     isProd1 = X[r][prodIndex1]
#     isProd2 = X[r][prodIndex2]  
#     if isProd1 and isProd2:
#         print(r, y[r])
#         for c in range(0, len(y[r])):
#             if y[r][c] == 1:
#                 print(c)
#                 if c != prodIndex1 and c != prodIndex2:
#                     print(map_index_productId[c])
#                     myset.add(map_index_productId[c])
# for prod in myset:
#     print(prod, ",")
        
#####################################################################################################################
#####################################################################################################################
# 3.  create matrices
    
#how long should x be?  each user should get n rows, where n is the number of products browsed.  each product gets to be Y once. 



XNumRows = 0
for k in map_customerId_productIdList:
    XNumRows += len(map_customerId_productIdList[k])

X = numpy.zeros(shape=(XNumRows, len(allProducts)))
yNonCat = numpy.zeros(XNumRows)

yProductIndicesList = list()

random.shuffle(allUsers)
rowIndex = -1
start_time2 = time.time()
for userIndex, userId in enumerate(allUsers):
    productsList = map_customerId_productIdList[userId]
    if len(productsList) < 2:
        continue
    productIndicesList = []
    for productId in productsList:
        productIndicesList.append(map_productId_index[productId])
#     print(userIndex, userId, productsList)
    for productIndexY in productIndicesList:
        rowIndex += 1
        yNonCat[rowIndex] = productIndexY
        shortList = productIndicesList[:]       # copy of list - remove productId in iteration
        shortList.remove(productIndexY)
        for productXIndex in shortList:
            X[rowIndex][productXIndex] += 1
#             try:
#                 tokens = map_productIndex_descriptionTokens[productXIndex]
#                 for token in tokens:
#                     tagIndex = map_tag_index[token]
#                     X[rowIndex][tagIndex + len(allProducts)] += 1
#             except:
#                 pass
print("elapsed time %g seconds" % (time.time() - start_time2))



# 
# for r in range(0, len(X)):
#     for c in range(0, len(X[0])):
#         if X[r][c] > 0:
#             print(r, c, X[r][c])

# build y

# now convert yProductIndicesList (now an array renamed to y) to one-hot encodings array
y = keras.utils.to_categorical(yNonCat)

# for r in range(0, len(y)):
#     for c in range(0, len(y[0])):
#         if y[r][c] > 0:
#             print(r, c, y[r][c])

# what next ... i think i have X and Y.
# now ... how to encode new browse data .... need product->index map and product->descriptionTags and tag->index
# to print - save X and Y as sparse matrices.  
# also save maps:
# # map_productId_index
# # map_productId_descriptionTokens
# # map_tag_index

# convert between sparse and dense?
# https://www.reddit.com/r/datascience/comments/5o2u06/how_to_pass_sparse_matrix_numpy_array_to_keras/
# save dict to file https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file

if not os.path.exists('dataMedium'):
    os.makedirs('dataMedium')
X_sparse_csr = sparse.csr_matrix(X)
y_sparse_csr = sparse.csr_matrix(y)
numpy.savez('dataMedium/X_sparse_csr_'+siteId, data=X_sparse_csr.data, indices=X_sparse_csr.indices, indptr=X_sparse_csr.indptr, shape=X_sparse_csr.shape)
numpy.savez('dataMedium/y_sparse_csr_'+siteId, data=y_sparse_csr.data, indices=y_sparse_csr.indices, indptr=y_sparse_csr.indptr, shape=y_sparse_csr.shape)
numpy.save('dataMedium/map_productId_index_'+siteId+'.npy', map_productId_index)
numpy.save('dataMedium/map_customerId_productIdList_'+siteId+'.npy', map_customerId_productIdList)
# numpy.save('dataShort/map_productId_descriptionTokens_'+siteId+'.npy', map_productId_descriptionTokens) 
# numpy.save('dataShort/map_tag_index_'+siteId+'.npy', map_tag_index)

# now what .... hmmm lets see how much memory python is using rn .... 23% of capacity ... hmm oh well.  
# next .... load this data into the NN code ... maybe i should just run now  while it's in memory... ?  or re-run from here - just takes 3 minutes
# yeah 

import numpy
import pandas
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

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# don't need specify input for internal layers: https://faroit.github.io/keras-docs/1.0.2/layers/core/
# define baseline model

X2 = X#[:10000,:]
y2 = y#[:10000,:]
def baseline_model():
    model = Sequential()
    model.add(Dense(len(X2[0]), input_dim=len(X2[0]), activation='relu'))
    model.add(Dropout(0.5))
#     model.add(Dense(int(len(X2[0])))) #makes loss very slightly less.  by 0.01 at epoch 11
    model.add(Dense(len(allProducts), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])    #optimizer was adam
    return model
    
filepath="weightsMedium-"+siteId+"-{epoch:02d}-{loss:.4f}.hdf5"  # define the checkpoint
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
print("# fit the model")
model = baseline_model()
model.fit(X2, y2, epochs=200, batch_size=20, callbacks=callbacks_list)  #starting at 4.44 - looking better after logout, reboot

# model.fit(X2, y2, epochs=200, batch_size=10, callbacks=callbacks_list)  #starting at 4.44 - looking better after logout, reboot


#####################################################################################################################################
#####################################################################################################################################
## getting recommendations
#
#

def printProductIds(list):
    print(str(list).replace("'", "").replace(" ", "").replace("[","").replace("]",""))

def sample(preds, temperature=1.0):
    if temperature == 99:
        return np.argmax(preds)
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def getRec(productIdsNew, diversity=1.0):
    Xnew = numpy.zeros(shape=(1, len(allProducts)))
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    prediction = model.predict(Xnew, verbose=1)[0]
    for productId in productIdsNew:
        productIndex = map_productId_index[productId]
        prediction[productIndex] = 0
    
    next_productIndex = sample(prediction, diversity)
    next_productId = map_index_productId[next_productIndex]
    return next_productId

def getRecs(productIds, n):
    for i in range(0, n):
        productIds.append(getRec(productIds))
    return productIds

    
printProductIds(getRecs(['OP-FCCT2646','OP-FCSS7-VST-K'], 100))
# 
#     pmax1 = -1
#     imax1 = 0
#     pmax2 = -1
#     imax2 = 0
#     pmax3 = -1
#     imax3 = 0
#     for i, p in enumerate(prediction[0]):
#         if p > pmax1 and map_index_productId[i] not in productIdsNew:
#             pmax3 = pmax2
#             pmax2 = pmax1
#             pmax1 = p
#             imax3 = imax2
#             imax2 = imax1
#             imax1 = i
#     print(map_productId_title[map_index_productId[imax1]], map_index_productId[imax1], ) #OP-PEBS2-CNS-BLS-K
#     print(map_productId_title[map_index_productId[imax2]], map_index_productId[imax2], ) #OP-PEBS2-CNS-BLS-K
#     print(map_productId_title[map_index_productId[imax3]], map_index_productId[imax3], ) #OP-PEBS2-CNS-BLS-K
#     print("")
    

productIdsNew = ['IP-ACC16GRK-GRY'] # 'OP-PEBS2-CNS-SPA-K','OP-PEBST5-BLS-K','IP-PEBS2-PORIII-GRY'] #stools
#map_productId_title

getRecs(['IP-ACC16GRK-GRY'], 100)

getRecs(['OP-FCCT2646','OP-FCSS7-VST-K','OP-PESS6MFT-PORIII-LGB-K','IP-PESS4-MAN-K','OP-FC4PC-PORIII'])

#now what .... take best weights file from checkpoint on aws -- save on laptop.  try to run prediction from loaded model. 
#then create endpoint in python to accept productIds and return recommendations

''' barstools: generate recommendation from these
OP-PEBS2-CNS-SPA-K
OP-PEBST5-BLS-K
IP-PEBS2-PORIII-GRY
'''

'''
actually this stuff ran after about 6 or 7 epochs earlier
>>> model.fit(X, y, epochs=100, batch_size=100, callbacks=callbacks_list)
Epoch 1/100
138800/139129 [============================>.] - ETA: 0s - loss: 3.1030 - acc: 0.2749Epoch 00000: loss did not improve
139129/139129 [==============================] - 14s - loss: 3.1030 - acc: 0.2749
Epoch 2/100
138700/139129 [============================>.] - ETA: 0s - loss: 2.9396 - acc: 0.3065Epoch 00001: loss improved from 3.05219 to 2.94035, saving model to weights-improvement-01-2.9403.hdf5
139129/139129 [==============================] - 13s - loss: 2.9403 - acc: 0.3063
Epoch 3/100
138800/139129 [============================>.] - ETA: 0s - loss: 2.8094 - acc: 0.3343Epoch 00002: loss improved from 2.94035 to 2.80944, saving model to weights-improvement-02-2.8094.hdf5
139129/139129 [==============================] - 13s - loss: 2.8094 - acc: 0.3343
Epoch 4/100
138800/139129 [============================>.] - ETA: 0s - loss: 2.6844 - acc: 0.3655Epoch 00003: loss improved from 2.80944 to 2.68456, saving model to weights-improvement-03-2.6846.hdf5
139129/139129 [==============================] - 13s - loss: 2.6846 - acc: 0.3655
Epoch 5/100
138700/139129 [============================>.] - ETA: 0s - loss: 2.5594 - acc: 0.4007Epoch 00004: loss improved from 2.68456 to 2.55947, saving model to weights-improvement-04-2.5595.hdf5
139129/139129 [==============================] - 13s - loss: 2.5595 - acc: 0.4007
Epoch 6/100
139000/139129 [============================>.] - ETA: 0s - loss: 2.4433 - acc: 0.4330Epoch 00005: loss improved from 2.55947 to 2.44341, saving model to weights-improvement-05-2.4434.hdf5
139129/139129 [==============================] - 13s - loss: 2.4434 - acc: 0.4330
Epoch 7/100
138700/139129 [============================>.] - ETA: 0s - loss: 2.3363 - acc: 0.4656Epoch 00006: loss improved from 2.44341 to 2.33670, saving model to weights-improvement-06-2.3367.hdf5
139129/139129 [==============================] - 13s - loss: 2.3367 - acc: 0.4657
'''

'''
>>> X2 = X#[:10000,:]
>>> y2 = y#[:10000,:]
>>> # don't need specify input for internal layers: https://faroit.github.io/keras-docs/1.0.2/layers/core/
... # define baseline model
... def baseline_model():
...     model = Sequential()
...     model.add(Dense(len(X2[0]), input_dim=len(X2[0]), activation='relu'))
... #     model.add(Dense(int(len(X2[0]))))
...     model.add(Dense(len(allProducts), activation='softmax'))
...     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
...     return model
...
>>>
>>> # define the checkpoint
... filepath="weightsMedium-improvement-{epoch:02d}-{loss:.4f}.hdf5"
>>> checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
>>> callbacks_list = [checkpoint]
>>> # fit the model
... model = baseline_model()
>>> model.fit(X, y, epochs=200, batch_size=100, callbacks=callbacks_list)
Epoch 1/200
138800/139129 [============================>.] - ETA: 0s - loss: 4.4048 - acc: 0.1981Epoch 00000: loss improved from inf to 4.40350, saving model to weightsMedium-improvement-00-4.4035.hdf5
139129/139129 [==============================] - 14s - loss: 4.4035 - acc: 0.1982
Epoch 2/200
138600/139129 [============================>.] - ETA: 0s - loss: 3.5418 - acc: 0.2337Epoch 00001: loss improved from 4.40350 to 3.54139, saving model to weightsMedium-improvement-01-3.5414.hdf5
139129/139129 [==============================] - 13s - loss: 3.5414 - acc: 0.2337
Epoch 3/200
139100/139129 [============================>.] - ETA: 0s - loss: 3.3241 - acc: 0.2459Epoch 00002: loss improved from 3.54139 to 3.32410, saving model to weightsMedium-improvement-02-3.3241.hdf5
139129/139129 [==============================] - 13s - loss: 3.3241 - acc: 0.2459
Epoch 4/200
139100/139129 [============================>.] - ETA: 0s - loss: 3.1950 - acc: 0.2577Epoch 00003: loss improved from 3.32410 to 3.19497, saving model to weightsMedium-improvement-03-3.1950.hdf5
139129/139129 [==============================] - 13s - loss: 3.1950 - acc: 0.2576
Epoch 5/200
138900/139129 [============================>.] - ETA: 0s - loss: 3.0846 - acc: 0.2726Epoch 00004: loss improved from 3.19497 to 3.08505, saving model to weightsMedium-improvement-04-3.0850.hdf5
139129/139129 [==============================] - 13s - loss: 3.0850 - acc: 0.2726
Epoch 6/200
139000/139129 [============================>.] - ETA: 0s - loss: 2.9736 - acc: 0.2931Epoch 00005: loss improved from 3.08505 to 2.97379, saving model to weightsMedium-improvement-05-2.9738.hdf5
139129/139129 [==============================] - 14s - loss: 2.9738 - acc: 0.2930
Epoch 7/200
139000/139129 [============================>.] - ETA: 0s - loss: 2.8620 - acc: 0.3184Epoch 00006: loss improved from 2.97379 to 2.86201, saving model to weightsMedium-improvement-06-2.8620.hdf5
139129/139129 [==============================] - 14s - loss: 2.8620 - acc: 0.3184
Epoch 8/200
138900/139129 [============================>.] - ETA: 0s - loss: 2.7566 - acc: 0.3432Epoch 00007: loss improved from 2.86201 to 2.75684, saving model to weightsMedium-improvement-07-2.7568.hdf5
139129/139129 [==============================] - 13s - loss: 2.7568 - acc: 0.3431
Epoch 9/200
138800/139129 [============================>.] - ETA: 0s - loss: 2.6429 - acc: 0.3741Epoch 00008: loss improved from 2.75684 to 2.64342, saving model to weightsMedium-improvement-08-2.6434.hdf5
139129/139129 [==============================] - 13s - loss: 2.6434 - acc: 0.3741
Epoch 10/200
139100/139129 [============================>.] - ETA: 0s - loss: 2.5350 - acc: 0.4060Epoch 00009: loss improved from 2.64342 to 2.53488, saving model to weightsMedium-improvement-09-2.5349.hdf5
139129/139129 [==============================] - 13s - loss: 2.5349 - acc: 0.4060
Epoch 11/200
139100/139129 [============================>.] - ETA: 0s - loss: 2.4265 - acc: 0.4392Epoch 00010: loss improved from 2.53488 to 2.42671, saving model to weightsMedium-improvement-10-2.4267.hdf5
139129/139129 [==============================] - 13s - loss: 2.4267 - acc: 0.4392
Epoch 12/200
138600/139129 [============================>.] - ETA: 0s - loss: 2.3257 - acc: 0.4680Epoch 00011: loss improved from 2.42671 to 2.32600, saving model to weightsMedium-improvement-11-2.3260.hdf5
139129/139129 [==============================] - 13s - loss: 2.3260 - acc: 0.4680
Epoch 13/200
138800/139129 [============================>.] - ETA: 0s - loss: 2.2324 - acc: 0.4959Epoch 00012: loss improved from 2.32600 to 2.23302, saving model to weightsMedium-improvement-12-2.2330.hdf5
139129/139129 [==============================] - 13s - loss: 2.2330 - acc: 0.4957
Epoch 14/200
138800/139129 [============================>.] - ETA: 0s - loss: 2.1464 - acc: 0.5198Epoch 00013: loss improved from 2.23302 to 2.14656, saving model to weightsMedium-improvement-13-2.1466.hdf5
139129/139129 [==============================] - 13s - loss: 2.1466 - acc: 0.5198
Epoch 15/200
138600/139129 [============================>.] - ETA: 0s - loss: 2.0754 - acc: 0.5395Epoch 00014: loss improved from 2.14656 to 2.07494, saving model to weightsMedium-improvement-14-2.0749.hdf5
139129/139129 [==============================] - 13s - loss: 2.0749 - acc: 0.5395
Epoch 16/200
138700/139129 [============================>.] - ETA: 0s - loss: 2.0086 - acc: 0.5570Epoch 00015: loss improved from 2.07494 to 2.00875, saving model to weightsMedium-improvement-15-2.0087.hdf5
139129/139129 [==============================] - 13s - loss: 2.0087 - acc: 0.5569
Epoch 17/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.9507 - acc: 0.5723Epoch 00016: loss improved from 2.00875 to 1.95037, saving model to weightsMedium-improvement-16-1.9504.hdf5
139129/139129 [==============================] - 13s - loss: 1.9504 - acc: 0.5724
Epoch 18/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.9015 - acc: 0.5846Epoch 00017: loss improved from 1.95037 to 1.90165, saving model to weightsMedium-improvement-17-1.9017.hdf5
139129/139129 [==============================] - 13s - loss: 1.9017 - acc: 0.5845
Epoch 19/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.8614 - acc: 0.5952Epoch 00018: loss improved from 1.90165 to 1.86172, saving model to weightsMedium-improvement-18-1.8617.hdf5
139129/139129 [==============================] - 13s - loss: 1.8617 - acc: 0.5952
Epoch 20/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.8221 - acc: 0.6045Epoch 00019: loss improved from 1.86172 to 1.82275, saving model to weightsMedium-improvement-19-1.8228.hdf5
139129/139129 [==============================] - 13s - loss: 1.8228 - acc: 0.6044
Epoch 21/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.7913 - acc: 0.6124Epoch 00020: loss improved from 1.82275 to 1.79132, saving model to weightsMedium-improvement-20-1.7913.hdf5
139129/139129 [==============================] - 13s - loss: 1.7913 - acc: 0.6124
Epoch 22/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.7638 - acc: 0.6199Epoch 00021: loss improved from 1.79132 to 1.76381, saving model to weightsMedium-improvement-21-1.7638.hdf5
139129/139129 [==============================] - 13s - loss: 1.7638 - acc: 0.6199
Epoch 23/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.7354 - acc: 0.6254Epoch 00022: loss improved from 1.76381 to 1.73556, saving model to weightsMedium-improvement-22-1.7356.hdf5
139129/139129 [==============================] - 13s - loss: 1.7356 - acc: 0.6253
Epoch 24/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.7153 - acc: 0.6313Epoch 00023: loss improved from 1.73556 to 1.71530, saving model to weightsMedium-improvement-23-1.7153.hdf5
139129/139129 [==============================] - 13s - loss: 1.7153 - acc: 0.6312
Epoch 25/200
139000/139129 [============================>.] - ETA: 0s - loss: 1.6983 - acc: 0.6351Epoch 00024: loss improved from 1.71530 to 1.69820, saving model to weightsMedium-improvement-24-1.6982.hdf5
139129/139129 [==============================] - 13s - loss: 1.6982 - acc: 0.6351
Epoch 26/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.6787 - acc: 0.6405Epoch 00025: loss improved from 1.69820 to 1.67867, saving model to weightsMedium-improvement-25-1.6787.hdf5
139129/139129 [==============================] - 13s - loss: 1.6787 - acc: 0.6406
Epoch 27/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.6696 - acc: 0.6432Epoch 00026: loss improved from 1.67867 to 1.66960, saving model to weightsMedium-improvement-26-1.6696.hdf5
139129/139129 [==============================] - 13s - loss: 1.6696 - acc: 0.6432
Epoch 28/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.6477 - acc: 0.6479Epoch 00027: loss improved from 1.66960 to 1.64828, saving model to weightsMedium-improvement-27-1.6483.hdf5
139129/139129 [==============================] - 13s - loss: 1.6483 - acc: 0.6478
Epoch 29/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.6358 - acc: 0.6501Epoch 00028: loss improved from 1.64828 to 1.63539, saving model to weightsMedium-improvement-28-1.6354.hdf5
139129/139129 [==============================] - 13s - loss: 1.6354 - acc: 0.6502
Epoch 30/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.6190 - acc: 0.6547Epoch 00029: loss improved from 1.63539 to 1.61900, saving model to weightsMedium-improvement-29-1.6190.hdf5
139129/139129 [==============================] - 13s - loss: 1.6190 - acc: 0.6547
Epoch 31/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.6104 - acc: 0.6576Epoch 00030: loss improved from 1.61900 to 1.61068, saving model to weightsMedium-improvement-30-1.6107.hdf5
139129/139129 [==============================] - 13s - loss: 1.6107 - acc: 0.6576
Epoch 32/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.6009 - acc: 0.6597Epoch 00031: loss improved from 1.61068 to 1.60097, saving model to weightsMedium-improvement-31-1.6010.hdf5
139129/139129 [==============================] - 13s - loss: 1.6010 - acc: 0.6597
Epoch 33/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.5873 - acc: 0.6626Epoch 00032: loss improved from 1.60097 to 1.58747, saving model to weightsMedium-improvement-32-1.5875.hdf5
139129/139129 [==============================] - 13s - loss: 1.5875 - acc: 0.6625
Epoch 34/200
139000/139129 [============================>.] - ETA: 0s - loss: 1.5776 - acc: 0.6664Epoch 00033: loss improved from 1.58747 to 1.57763, saving model to weightsMedium-improvement-33-1.5776.hdf5
139129/139129 [==============================] - 13s - loss: 1.5776 - acc: 0.6664
Epoch 35/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.5696 - acc: 0.6675Epoch 00034: loss improved from 1.57763 to 1.56976, saving model to weightsMedium-improvement-34-1.5698.hdf5
139129/139129 [==============================] - 13s - loss: 1.5698 - acc: 0.6675
Epoch 36/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.5629 - acc: 0.6703Epoch 00035: loss improved from 1.56976 to 1.56264, saving model to weightsMedium-improvement-35-1.5626.hdf5
139129/139129 [==============================] - 13s - loss: 1.5626 - acc: 0.6703
Epoch 37/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5590 - acc: 0.6718Epoch 00036: loss improved from 1.56264 to 1.55955, saving model to weightsMedium-improvement-36-1.5596.hdf5
139129/139129 [==============================] - 14s - loss: 1.5596 - acc: 0.6717
Epoch 38/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.5499 - acc: 0.6743Epoch 00037: loss improved from 1.55955 to 1.55009, saving model to weightsMedium-improvement-37-1.5501.hdf5
139129/139129 [==============================] - 14s - loss: 1.5501 - acc: 0.6744
Epoch 39/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5483 - acc: 0.6755Epoch 00038: loss improved from 1.55009 to 1.54836, saving model to weightsMedium-improvement-38-1.5484.hdf5
139129/139129 [==============================] - 13s - loss: 1.5484 - acc: 0.6755
Epoch 40/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5378 - acc: 0.6770Epoch 00039: loss improved from 1.54836 to 1.53795, saving model to weightsMedium-improvement-39-1.5380.hdf5
139129/139129 [==============================] - 13s - loss: 1.5380 - acc: 0.6770
Epoch 41/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.5262 - acc: 0.6798Epoch 00040: loss improved from 1.53795 to 1.52673, saving model to weightsMedium-improvement-40-1.5267.hdf5
139129/139129 [==============================] - 13s - loss: 1.5267 - acc: 0.6797
Epoch 42/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5208 - acc: 0.6807Epoch 00041: loss improved from 1.52673 to 1.52067, saving model to weightsMedium-improvement-41-1.5207.hdf5
139129/139129 [==============================] - 14s - loss: 1.5207 - acc: 0.6808
Epoch 43/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5211 - acc: 0.6820Epoch 00042: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.5218 - acc: 0.6819
Epoch 44/200
139000/139129 [============================>.] - ETA: 0s - loss: 1.5178 - acc: 0.6835Epoch 00043: loss improved from 1.52067 to 1.51754, saving model to weightsMedium-improvement-43-1.5175.hdf5
139129/139129 [==============================] - 13s - loss: 1.5175 - acc: 0.6835
Epoch 45/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.5095 - acc: 0.6847Epoch 00044: loss improved from 1.51754 to 1.50968, saving model to weightsMedium-improvement-44-1.5097.hdf5
139129/139129 [==============================] - 14s - loss: 1.5097 - acc: 0.6846
Epoch 46/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.5025 - acc: 0.6869Epoch 00045: loss improved from 1.50968 to 1.50253, saving model to weightsMedium-improvement-45-1.5025.hdf5
139129/139129 [==============================] - 13s - loss: 1.5025 - acc: 0.6869
Epoch 47/200
138600/139129 [============================>.] - ETA: 0s - loss: 1.5157 - acc: 0.6868Epoch 00046: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.5155 - acc: 0.6868
Epoch 48/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.5130 - acc: 0.6882Epoch 00047: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.5129 - acc: 0.6882
Epoch 49/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.5026 - acc: 0.6899Epoch 00048: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.5027 - acc: 0.6898
Epoch 50/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.5063 - acc: 0.6914Epoch 00049: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.5067 - acc: 0.6913
Epoch 51/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.5133 - acc: 0.6902Epoch 00050: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.5130 - acc: 0.6903
Epoch 52/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.5023 - acc: 0.6916Epoch 00051: loss improved from 1.50253 to 1.50207, saving model to weightsMedium-improvement-51-1.5021.hdf5
139129/139129 [==============================] - 13s - loss: 1.5021 - acc: 0.6917
Epoch 53/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.4885 - acc: 0.6947Epoch 00052: loss improved from 1.50207 to 1.48833, saving model to weightsMedium-improvement-52-1.4883.hdf5
139129/139129 [==============================] - 13s - loss: 1.4883 - acc: 0.6946
Epoch 54/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.4891 - acc: 0.6954Epoch 00053: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.4890 - acc: 0.6955
Epoch 55/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.4930 - acc: 0.6948Epoch 00054: loss did not improve
139129/139129 [==============================] - 13s - loss: 1.4929 - acc: 0.6948
Epoch 56/200
138800/139129 [============================>.] - ETA: 0s - loss: 1.4859 - acc: 0.6962Epoch 00055: loss improved from 1.48833 to 1.48634, saving model to weightsMedium-improvement-55-1.4863.hdf5
139129/139129 [==============================] - 14s - loss: 1.4863 - acc: 0.6961
Epoch 57/200
138700/139129 [============================>.] - ETA: 0s - loss: 1.4854 - acc: 0.6967Epoch 00056: loss improved from 1.48634 to 1.48575, saving model to weightsMedium-improvement-56-1.4857.hdf5
139129/139129 [==============================] - 14s - loss: 1.4857 - acc: 0.6967
Epoch 58/200
138900/139129 [============================>.] - ETA: 0s - loss: 1.4880 - acc: 0.6979Epoch 00057: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.4883 - acc: 0.6978
Epoch 59/200
139000/139129 [============================>.] - ETA: 0s - loss: 1.4717 - acc: 0.7001Epoch 00058: loss improved from 1.48575 to 1.47146, saving model to weightsMedium-improvement-58-1.4715.hdf5
139129/139129 [==============================] - 14s - loss: 1.4715 - acc: 0.7001
Epoch 60/200
139000/139129 [============================>.] - ETA: 0s - loss: 1.4738 - acc: 0.6998Epoch 00059: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.4740 - acc: 0.6998
Epoch 61/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.4773 - acc: 0.7009Epoch 00060: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.4772 - acc: 0.7009
Epoch 62/200
139100/139129 [============================>.] - ETA: 0s - loss: 1.4793 - acc: 0.7008Epoch 00061: loss did not improve
139129/139129 [==============================] - 14s - loss: 1.4791 - acc: 0.7008
Epoch 63/200
 38900/139129 [=======>......................] - ETA: 10s - loss: 1.4514 - acc: 0.7042^CTraceback (most recent call last):
 '''

estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=100, verbose=1)
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)

#what is this doing?  why the model generator function?  try to just fit the model like normal.  do this on aws
start_time2 = time.time()
results = cross_val_score(estimator, X2, y2, cv=kfold)
print("elapsed time %g seconds" % (time.time() - start_time2))
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))








#  abt 3 minutes total
# # #build X
# # #insert productIds and tagIds (description token ids) into X
# for userIndex, userId in enumerate(allUsers):
#     productsList = map_customerId_productIdList[userId]
#     productIndicesAr = numpy.zeros(len(productsList), dtype=numpy.int)
#     for i in range(0, len(productIndicesAr)):
#         productIndicesAr[i] = map_productId_index[productsList[i]]    
#     print(userIndex, userId, productsList)
#     print(userIndex, userId, productIndicesAr)
#     for productIndicesArI in range(0, len(productIndicesAr)):
#         productIndexY = productIndicesAr[productIndicesArI]
#         rowIndex += 1
#         y[rowIndex] = productIndexY
#         shortList = numpy.zeros(len(productIndicesAr) - 1, dtype=numpy.int)
#         print("row", rowIndex, shortList)
#         i = 0
#         while i < len(shortList):
#             if i == productIndicesArI:
#                 i += 1
#                 continue
#             shortList[i] = productIndicesAr[i]
#             i += 1
#         for productXIndex in shortList:
#             X[rowIndex][productXIndex] += 1
#             try:
#                 tokens = map_productIndex_descriptionTokens[productXIndex]
#                 for token in tokens:
#                     tagIndex = map_tag_index[token]
#                     X[rowIndex][tagIndex + len(allProducts)] += 1
#             except:
#                 pass
#alternate:
       
# x = 2
# print(x)                
#    8:08:25 - a little less than 3 minutes
# for userIndex, userId in enumerate(allUsers):
#     productsList = map_customerId_productIdList[userId]
#     print(userIndex, userId, productsList)
#     for productIdY in productsList:
#         rowIndex += 1
#         y[rowIndex] = map_productId_index[productIdY]
#         shortList = productsList[:]       # copy of list - remove productId in iteration
#         shortList.remove(productIdY)
#         for productIdX in shortList:
#             productXIndex = map_productId_index[productIdX]
# #             print(productXIndex, productIdX)
#             X[rowIndex][productXIndex] += 1
#             try:
#                 tokens = map_productId_descriptionTokens[productIdX]
#                 for token in tokens:
#                     tagIndex = map_tag_index[token]
#                     X[rowIndex][tagIndex + len(allProducts)] += 1
#             except:
#                 pass

# 8:17:00 - took about 2:40    
# XNumRows = 0
# for k in map_customerId_productIdList:
#     XNumRows += len(map_customerId_productIdList[k])
# 
# X = numpy.zeros(shape=(XNumRows, len(allProducts) + len(allTagsSet)))
# y = numpy.zeros(XNumRows)
# 
# yProductIndicesList = list()
# 
# rowIndex = -1
# for userIndex, userId in enumerate(allUsers):
#     productsList = map_customerId_productIdList[userId]
#     productIndicesList = []
#     for productId in productsList:
#         productIndicesList.append(map_productId_index[productId])
#     print(userIndex, userId, productsList)
#     for productIndexY in productIndicesList:
#         rowIndex += 1
#         y[rowIndex] = productIndexY
#         shortList = productIndicesList[:]       # copy of list - remove productId in iteration
#         shortList.remove(productIndexY)
#         for productXIndex in shortList:
#             X[rowIndex][productXIndex] += 1
#             try:
#                 tokens = map_productIndex_descriptionTokens[productXIndex]
#                 for token in tokens:
#                     tagIndex = map_tag_index[token]
#                     X[rowIndex][tagIndex + len(allProducts)] += 1
#             except:
#                 pass
#
####
#
#
#
#

## sorted(data, key=lambda tup: tup[1])