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
#####################################################################################################################
#####################################################################################################################
# 3.  create matrices
    
#how long should x be?  each user should get n rows, where n is the number of products browsed.  each product gets to be Y once. 

# 8:17:00 - took about 2:40 on mac.  20 seconds faster on aws -- although sometimes much longer! 
# 
# XNumRows = 0
# for k in map_customerId_productIdList:
#     XNumRows += len(map_customerId_productIdList[k])
# 
# X = numpy.zeros(shape=(XNumRows, len(allProducts) + len(allTagsSet)))
# yNonCat = numpy.zeros(XNumRows)

#instead of making len(productList) training samples per user, just make 1.  the last product in the list.  hopefully is chronological?

XNumRows = len(map_customerId_productIdList)

X = numpy.zeros(shape=(XNumRows, len(allProducts)))
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
    productIndexY = productIndicesList[-1]
    rowIndex += 1
    yNonCat[rowIndex] = productIndexY
    shortList = productIndicesList[:-1]       # copy of list - remove last productIndex (used for y)
    for productXIndex in shortList:
        X[rowIndex][productXIndex] += 1
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

if not os.path.exists('dataShort'):
    os.makedirs('dataShort')
    
X_sparse_csr = sparse.csr_matrix(X)
y_sparse_csr = sparse.csr_matrix(y)

numpy.savez('dataShort/X_sparse_csr', data=X_sparse_csr.data, indices=X_sparse_csr.indices, indptr=X_sparse_csr.indptr, shape=X_sparse_csr.shape)
numpy.savez('dataShort/y_sparse_csr', data=y_sparse_csr.data, indices=y_sparse_csr.indices, indptr=y_sparse_csr.indptr, shape=y_sparse_csr.shape)
numpy.save('dataShort/map_productId_index.npy', map_productId_index) 
# numpy.save('dataShort/map_productId_descriptionTokens.npy', map_productId_descriptionTokens) 
# numpy.save('dataShort/map_tag_index.npy', map_tag_index)

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

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
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
