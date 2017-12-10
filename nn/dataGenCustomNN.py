from __future__ import print_function
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy import sparse
from collections import defaultdict

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


#### TODO
## - write/read data to file (like i'm doing.  stop re-processing data per run.     need to run discrete file on aws cos bad connection.  
## write some mac thing to reset wireless connection if internet lost
## tensorflow taking long time ... compile on machine?  trying rn w/ very small hidden layer.  but still takes 2+ minutes to prepare data. ran for 30 minutes so far ... nothing :( even with tiny hidden layer D:  need embedding??????
## start w/ smaller data?  no time to get new site ... just shrink existing data?  
## wtf why saying elapsed time - 263 seconds D:
## will adding dropout speed it up?

#####################################################################################################################
# 1.  get item features.  build tagIndex->tag map

stemmer = PorterStemmer()

stopwords = ['qualiti', 'live', 'use', 'exist', 'allow', 'ad', 'start', 'make', 'way', 'rst', 'll', 're', 'd', 've', 's', 't', 'br', 'li', 'nbsp', 'p', 'span', 'div', 'ul', 'ol', 'includes','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

def get_words(doc):
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))     # replace punctuation with space
    doc = doc.translate(replace_punctuation)
    tokens = doc.split()                                                                            # split into tokens by white space
    tokens = [word for word in tokens if (word.isalpha() and word not in stopwords)]       # remove remaining tokens that are not alphabetic
    tokens = [stemmer.stem(word.lower()) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

map_productId_descriptionTokens = {}
path = "fordocker/products_production_siteId_=" + siteId + "/*.csv"
nonascii = bytearray(range(0x80, 0x100))
for fname in glob.glob(path):
    print(fname)
    with open(fname, mode='rb') as infile:
        next(infile)
        for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
            line = line.translate(None, nonascii)
            line = str(line,'utf-8')
            for row in csv.reader([line]):
                k,v  = row[0], get_words(row[2]) + get_words(row[3])[:13] 
                map_productId_descriptionTokens[k] = v
shared = {}
allTagsSet = {}
# get starter description
for key in map_productId_descriptionTokens:
    shared = set(map_productId_descriptionTokens[key])
    allTagsSet = set(map_productId_descriptionTokens[key])
    break

for key in map_productId_descriptionTokens:
    shared = shared.intersection(map_productId_descriptionTokens[key])
    allTagsSet = allTagsSet.union(map_productId_descriptionTokens[key])

#done making item features!!!!!!   what next?  collect all tags into a single list/vector.  now, made maps!  seeeeee

allTagsSet = list(allTagsSet)

map_index_tag = dict(enumerate(allTagsSet))
map_tag_index = {x:i for i,x in enumerate(allTagsSet)}


#####################################################################################################################
#####################################################################################################################
# 2. get browse lists


allProducts = set()
map_customerId_productIdList = defaultdict(list)

# if True:
#     fname = "fordocker/browse_production_siteId_=35569/part-r-00016-55b1cd2d-d2c7-43dc-ac2a-da953f82d47b.csv"
path = "fordocker/browse_production_siteId_=35569/*.csv"
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
                allProducts.add(product_id)
                map_customerId_productIdList[customer_id].append(product_id)

#ok now i have lists of products per user ... now what .... get set of users and set of products.  for amounts.

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


map_productIndex_descriptionTokens = {}

for productId in map_productId_descriptionTokens:
    try:
        productIndex = map_productId_index[productId]
#         print(productId, productIndex)
        map_productIndex_descriptionTokens[productIndex] = map_productId_descriptionTokens[productId]
    except:
        pass
    

#####################################################################################################################
#####################################################################################################################
# 3.  create matrices
    
#how long should x be?  each user should get n rows, where n is the number of products browsed.  each product gets to be Y once. 

# 8:17:00 - took about 2:40    
# 12:56:20 - finished some time b4 12:59:40
XNumRows = 0
for k in map_customerId_productIdList:
    XNumRows += len(map_customerId_productIdList[k])

X = numpy.zeros(shape=(XNumRows, len(allProducts) + len(allTagsSet)))
yNonCat = numpy.zeros(XNumRows)

yProductIndicesList = list()

rowIndex = -1
start_time2 = time.time()
for userIndex, userId in enumerate(allUsers):
    productsList = map_customerId_productIdList[userId]
    productIndicesList = []
    for productId in productsList:
        productIndicesList.append(map_productId_index[productId])
    print(userIndex, userId, productsList)
    for productIndexY in productIndicesList:
        rowIndex += 1
        yNonCat[rowIndex] = productIndexY
        shortList = productIndicesList[:]       # copy of list - remove productId in iteration
        shortList.remove(productIndexY)
        for productXIndex in shortList:
            X[rowIndex][productXIndex] += 1
            try:
                tokens = map_productIndex_descriptionTokens[productXIndex]
                for token in tokens:
                    tagIndex = map_tag_index[token]
                    X[rowIndex][tagIndex + len(allProducts)] += 1
            except Exception:
                pass
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

if not os.path.exists('data'):
    os.makedirs('data')
    
X_sparse_csr = sparse.csr_matrix(X)
y_sparse_csr = sparse.csr_matrix(y)

numpy.savez('data/X_sparse_csr', data=X_sparse_csr.data, indices=X_sparse_csr.indices, indptr=X_sparse_csr.indptr, shape=X_sparse_csr.shape)
numpy.savez('data/y_sparse_csr', data=y_sparse_csr.data, indices=y_sparse_csr.indices, indptr=y_sparse_csr.indptr, shape=y_sparse_csr.shape)
numpy.save('data/map_productId_index.npy', map_productId_index) 
numpy.save('data/map_productId_descriptionTokens.npy', map_productId_descriptionTokens) 
numpy.save('data/map_tag_index.npy', map_tag_index) 

# now what .... hmmm lets see how much memory python is using rn .... 23% of capacity ... hmm oh well.  
# next .... load this data into the NN code ... maybe i should just run now  while it's in memory... ?  or re-run from here - just takes 3 minutes
# yeah 

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

## copied from dataGenCustomNNmedium

X2 = X#[:10000,:]
y2 = y#[:10000,:]
# don't need specify input for internal layers: https://faroit.github.io/keras-docs/1.0.2/layers/core/
# define baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(len(X2[0]), input_dim=len(X2[0]), activation='relu'))
    model.add(Dense(int(len(X2[0])))) #makes loss very slightly less.  by 0.01 at epoch 11
    model.add(Dense(len(allProducts), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#>>> len(X[0])
#2106

# define the checkpoint
filepath="weightsWithItemFeatures-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
modelWithItemFeatures = baseline_model()
modelWithItemFeatures.fit(X2, y2, epochs=100, batch_size=100, callbacks=callbacks_list)
'''
>>> # define the checkpoint
... filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
>>> checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
>>> callbacks_list = [checkpoint]
>>> # fit the model
... model = baseline_model()
>>> model.fit(X2, y2, epochs=100, batch_size=100, callbacks=callbacks_list)
Epoch 1/100
139000/139129 [============================>.] - ETA: 0s - loss: 4.7046 - acc: 0.2050Epoch 00000: loss improved from inf to 4.70418, saving model to weights-improvement-00-4.7042.hdf5
139129/139129 [==============================] - 21s - loss: 4.7042 - acc: 0.2051
Epoch 2/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.9330 - acc: 0.2285Epoch 00001: loss improved from 4.70418 to 3.93310, saving model to weights-improvement-01-3.9331.hdf5
139129/139129 [==============================] - 21s - loss: 3.9331 - acc: 0.2284
Epoch 3/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.7127 - acc: 0.2359Epoch 00002: loss improved from 3.93310 to 3.71318, saving model to weights-improvement-02-3.7132.hdf5
139129/139129 [==============================] - 21s - loss: 3.7132 - acc: 0.2358
Epoch 4/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.5708 - acc: 0.2408Epoch 00003: loss improved from 3.71318 to 3.57103, saving model to weights-improvement-03-3.5710.hdf5
139129/139129 [==============================] - 21s - loss: 3.5710 - acc: 0.2408
Epoch 5/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.4682 - acc: 0.2474Epoch 00004: loss improved from 3.57103 to 3.46837, saving model to weights-improvement-04-3.4684.hdf5
139129/139129 [==============================] - 21s - loss: 3.4684 - acc: 0.2474
Epoch 6/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.3870 - acc: 0.2505Epoch 00005: loss improved from 3.46837 to 3.38692, saving model to weights-improvement-05-3.3869.hdf5
139129/139129 [==============================] - 21s - loss: 3.3869 - acc: 0.2505
Epoch 7/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.3116 - acc: 0.2565Epoch 00006: loss improved from 3.38692 to 3.31159, saving model to weights-improvement-06-3.3116.hdf5
139129/139129 [==============================] - 21s - loss: 3.3116 - acc: 0.2566
Epoch 8/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.2438 - acc: 0.2604Epoch 00007: loss improved from 3.31159 to 3.24386, saving model to weights-improvement-07-3.2439.hdf5
139129/139129 [==============================] - 21s - loss: 3.2439 - acc: 0.2604
Epoch 9/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.1838 - acc: 0.2671Epoch 00008: loss improved from 3.24386 to 3.18389, saving model to weights-improvement-08-3.1839.hdf5
139129/139129 [==============================] - 21s - loss: 3.1839 - acc: 0.2671
Epoch 10/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.1361 - acc: 0.2741Epoch 00009: loss improved from 3.18389 to 3.13571, saving model to weights-improvement-09-3.1357.hdf5
139129/139129 [==============================] - 21s - loss: 3.1357 - acc: 0.2742
Epoch 11/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.0880 - acc: 0.2810Epoch 00010: loss improved from 3.13571 to 3.08808, saving model to weights-improvement-10-3.0881.hdf5
139129/139129 [==============================] - 21s - loss: 3.0881 - acc: 0.2810
Epoch 12/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.0422 - acc: 0.2892Epoch 00011: loss improved from 3.08808 to 3.04187, saving model to weights-improvement-11-3.0419.hdf5
139129/139129 [==============================] - 21s - loss: 3.0419 - acc: 0.2893
Epoch 13/100
138900/139129 [============================>.] - ETA: 0s - loss: 3.0051 - acc: 0.2959Epoch 00012: loss improved from 3.04187 to 3.00518, saving model to weights-improvement-12-3.0052.hdf5
139129/139129 [==============================] - 21s - loss: 3.0052 - acc: 0.2959
Epoch 14/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.9644 - acc: 0.3026Epoch 00013: loss improved from 3.00518 to 2.96432, saving model to weights-improvement-13-2.9643.hdf5
139129/139129 [==============================] - 21s - loss: 2.9643 - acc: 0.3026
Epoch 15/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.9270 - acc: 0.3099Epoch 00014: loss improved from 2.96432 to 2.92742, saving model to weights-improvement-14-2.9274.hdf5
139129/139129 [==============================] - 21s - loss: 2.9274 - acc: 0.3099
Epoch 16/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.8951 - acc: 0.3186Epoch 00015: loss improved from 2.92742 to 2.89516, saving model to weights-improvement-15-2.8952.hdf5
139129/139129 [==============================] - 21s - loss: 2.8952 - acc: 0.3186
Epoch 17/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.8584 - acc: 0.3260Epoch 00016: loss improved from 2.89516 to 2.85875, saving model to weights-improvement-16-2.8587.hdf5
139129/139129 [==============================] - 21s - loss: 2.8587 - acc: 0.3260
Epoch 18/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.8325 - acc: 0.3316Epoch 00017: loss improved from 2.85875 to 2.83228, saving model to weights-improvement-17-2.8323.hdf5
139129/139129 [==============================] - 21s - loss: 2.8323 - acc: 0.3316
Epoch 19/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.8039 - acc: 0.3412Epoch 00018: loss improved from 2.83228 to 2.80437, saving model to weights-improvement-18-2.8044.hdf5
139129/139129 [==============================] - 21s - loss: 2.8044 - acc: 0.3411
Epoch 20/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.7772 - acc: 0.3452Epoch 00019: loss improved from 2.80437 to 2.77746, saving model to weights-improvement-19-2.7775.hdf5
139129/139129 [==============================] - 21s - loss: 2.7775 - acc: 0.3453
Epoch 21/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.7516 - acc: 0.3535Epoch 00020: loss improved from 2.77746 to 2.75168, saving model to weights-improvement-20-2.7517.hdf5
139129/139129 [==============================] - 21s - loss: 2.7517 - acc: 0.3535
Epoch 22/100
138800/139129 [============================>.] - ETA: 0s - loss: 2.7261 - acc: 0.3601Epoch 00021: loss improved from 2.75168 to 2.72638, saving model to weights-improvement-21-2.7264.hdf5
139129/139129 [==============================] - 21s - loss: 2.7264 - acc: 0.3602
Epoch 23/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.7020 - acc: 0.3665Epoch 00022: loss improved from 2.72638 to 2.70224, saving model to weights-improvement-22-2.7022.hdf5
139129/139129 [==============================] - 21s - loss: 2.7022 - acc: 0.3664
Epoch 24/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.6829 - acc: 0.3727Epoch 00023: loss improved from 2.70224 to 2.68337, saving model to weights-improvement-23-2.6834.hdf5
139129/139129 [==============================] - 21s - loss: 2.6834 - acc: 0.3726
Epoch 25/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.6615 - acc: 0.3786Epoch 00024: loss improved from 2.68337 to 2.66123, saving model to weights-improvement-24-2.6612.hdf5
139129/139129 [==============================] - 21s - loss: 2.6612 - acc: 0.3787
Epoch 26/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.6449 - acc: 0.3846Epoch 00025: loss improved from 2.66123 to 2.64525, saving model to weights-improvement-25-2.6453.hdf5
139129/139129 [==============================] - 21s - loss: 2.6453 - acc: 0.3845
Epoch 27/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.6226 - acc: 0.3883Epoch 00026: loss improved from 2.64525 to 2.62279, saving model to weights-improvement-26-2.6228.hdf5
139129/139129 [==============================] - 21s - loss: 2.6228 - acc: 0.3883
Epoch 28/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.6023 - acc: 0.3948Epoch 00027: loss improved from 2.62279 to 2.60269, saving model to weights-improvement-27-2.6027.hdf5
139129/139129 [==============================] - 21s - loss: 2.6027 - acc: 0.3947
Epoch 29/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5886 - acc: 0.4014Epoch 00028: loss improved from 2.60269 to 2.58830, saving model to weights-improvement-28-2.5883.hdf5
139129/139129 [==============================] - 21s - loss: 2.5883 - acc: 0.4014
Epoch 30/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5709 - acc: 0.4053Epoch 00029: loss improved from 2.58830 to 2.57114, saving model to weights-improvement-29-2.5711.hdf5
139129/139129 [==============================] - 21s - loss: 2.5711 - acc: 0.4053
Epoch 31/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5535 - acc: 0.4091Epoch 00030: loss improved from 2.57114 to 2.55344, saving model to weights-improvement-30-2.5534.hdf5
139129/139129 [==============================] - 21s - loss: 2.5534 - acc: 0.4091
Epoch 32/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5390 - acc: 0.4142Epoch 00031: loss improved from 2.55344 to 2.53876, saving model to weights-improvement-31-2.5388.hdf5
139129/139129 [==============================] - 21s - loss: 2.5388 - acc: 0.4142
Epoch 33/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5224 - acc: 0.4194Epoch 00032: loss improved from 2.53876 to 2.52247, saving model to weights-improvement-32-2.5225.hdf5
139129/139129 [==============================] - 21s - loss: 2.5225 - acc: 0.4194
Epoch 34/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.5038 - acc: 0.4234Epoch 00033: loss improved from 2.52247 to 2.50363, saving model to weights-improvement-33-2.5036.hdf5
139129/139129 [==============================] - 21s - loss: 2.5036 - acc: 0.4235
Epoch 35/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.4907 - acc: 0.4293Epoch 00034: loss improved from 2.50363 to 2.49062, saving model to weights-improvement-34-2.4906.hdf5
139129/139129 [==============================] - 21s - loss: 2.4906 - acc: 0.4293
Epoch 36/100
138900/139129 [============================>.] - ETA: 0s - loss: 2.4772 - acc: 0.4334Epoch 00035: loss improved from 2.49062 to 2.47692, saving model to weights-improvement-35-2.4769.hdf5
139129/139129 [==============================] - 21s - loss: 2.4769 - acc: 0.4335
'''




# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=len(X[0]), activation='relu'))
#     model.add(Dense(len(allProducts), activation='softmax'))
    model.add(Dense(len(allProducts), activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model = baseline_model()
model.fit(X, y, epochs=100, batch_size=100, callbacks=callbacks_list)



estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#what is this doing?  why the model generator function?  try to just fit the model like normal.  do this on aws

results = cross_val_score(estimator, X, y, cv=kfold)
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
