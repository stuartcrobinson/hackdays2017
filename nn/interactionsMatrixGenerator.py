from scipy import sparse
from collections import defaultdict

import glob
import csv
import sys  
import io
import numpy
import random

# later fix this to look at all browse files
# customer_id,contact_id,product_id,event_date,url
# path = "fordocker/browse_production_siteId_=35569/*.csv"
# map_customerId_productIdList= {}
# for fname in glob.glob(path):

# allUsers = {}
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

interactionsAll = numpy.zeros(shape=(len(allUsers),len(allProducts)))
interactionsTest = numpy.zeros(shape=(len(allUsers),len(allProducts)))
interactionsTrain = numpy.zeros(shape=(len(allUsers),len(allProducts)))

for userIndex, userId in enumerate(allUsers):
    productsList = map_customerId_productIdList[userId]
#     print(userIndex,  " ", userId) 
#     print(productsList)
    for productId in productsList:
#         print(productId)
        productIndex = map_productId_index[productId]
        interactionsAll[userIndex][productIndex] += 1
        #now put in test or train.  get random true 80%
        if random.randint(1,100) > 20:
            interactionsTrain[userIndex][productIndex] += 1   
        else:
            interactionsTest[userIndex][productIndex] += 1

#now convert to coo sparse matrix

interactionsAll_sparse_coo_matrix = sparse.coo_matrix(interactionsAll)
interactionsTest_sparse_coo_matrix = sparse.coo_matrix(interactionsTest)
interactionsTrain_sparse_coo_matrix = sparse.coo_matrix(interactionsTrain)

print('')
print(interactionsAll_sparse_coo_matrix.getnnz())
print(interactionsTest_sparse_coo_matrix.getnnz())
print(interactionsTrain_sparse_coo_matrix.getnnz())

def save_sparse_coo(filename,array):
    numpy.savez(filename, data=array.data, row=array.row, col=array.col, shape=array.shape )
                         
def load_sparse_coo(filename):
    loader = numpy.load(filename)
    return coo_matrix((  loader['data'], (loader['row'], loader['col'])),  shape = loader['shape'])

save_sparse_coo('fordocker/interactionsAll_sparse_coo_matrix', interactionsAll_sparse_coo_matrix)
save_sparse_coo('fordocker/interactionsTest_sparse_coo_matrix', interactionsTest_sparse_coo_matrix)
save_sparse_coo('fordocker/interactionsTrain_sparse_coo_matrix', interactionsTrain_sparse_coo_matrix)


# 
# numpy.savez('interactionsAll_sparse_coo_matrix', matrix=interactionsAll_sparse_coo_matrix)
# numpy.savez('interactionsTest_sparse_coo_matrix', matrix=interactionsTest_sparse_coo_matrix)
# numpy.savez('interactionsTrain_sparse_coo_matrix', matrix=interactionsTrain_sparse_coo_matrix)
#         




# 
# ####no
# path = "fordocker/products_production_siteId_=35569/*.csv"
# d = {}
# for fname in glob.glob(path):
#     print(fname)
#     nonascii = bytearray(range(0x80, 0x100))
# #     with open(fname,'rb') as infile, open('fordocker/d_parsed.csv','wb') as outfile:
#     with open(fname,'rb') as infile:
#         next(infile)
#         for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
#             line = line.translate(None, nonascii)
#             line = str(line,'utf-8')
#             for row in csv.reader([line]):
#                 print(row[0])
#                 k,v  = row[0], clean(row[2] + " " + row[3])[:-1]
# #                 v = row[3]
#                 d[k] = v
# ####no




# ### orig file reading method - getting ascii problems.  read bites and deal w/ like item features code
# if True:
#     fname = "fordocker/browse_production_siteId_=35569/part-r-00016-55b1cd2d-d2c7-43dc-ac2a-da953f82d47b.csv"
# path = "fordocker/browse_production_siteId_=35569/*.csv"
# for fname in glob.glob(path):
#     print(fname)
#     with open(fname, mode='r') as infile:
#         reader = csv.reader(infile)
#         next(reader, None)
#         for row in reader:
#             # print(row)
#             customer_id = row[0]
#             product_id = row[2]
#             allProducts.add(product_id)
#             map_customerId_productIdList[customer_id].append(product_id)



