from scipy import sparse
from collections import defaultdict

import glob
import csv
import sys  
import io
import numpy
import random
'''

write:
 - X and Y matrices and 
 - productIdIndex -> productId map
 - tagIndex -> tag

X: - first n columns are productIds, the rest are tagIds
0 0 0 0 1 0 0 1 1 0 0 2 0 1 1 0 0 0 3 0 
0 0 0 1 0 0 0 1 1 0 0 2 0 1 6 0 0 0 3 0 
0 0 0 0 1 0 0 1 4 0 0 2 0 1 1 0 0 0 3 0 

Y: - one-hot encoding of classifications - columns are productIds
[0 0 0 0 0 0 1 0 0 0]
[0 0 0 0 1 0 0 0 0 0]
[0 0 0 0 0 1 0 0 0 0]


so when sending in new browse data to get new recs ... create X (for predict)

X: - first n columns are productIds, the rest are tagIds
0 0 0 0 1 0 0 1 1 0 0 2 0 1 1 0 0 0 3 0 



import numpy as np

# Save
dictionary = {'hello':'world'}
np.save('my_file.npy', dictionary) 

# Load
read_dictionary = np.load('my_file.npy').item()
print(read_dictionary['hello']) # displays "world"

'''

# later fix this to look at all browse files
# customer_id,contact_id,product_id,event_date,url
# path = "fordocker/browse_production_siteId_=35569/*.csv"
# map_customerId_productIdList= {}
# for fname in glob.glob(path):

# allUsers = {}
allProducts = set()

map_customerId_productIdList = defaultdict(list)

siteId = 35569

# if True:
#     fname = "fordocker/browse_production_siteId_=35569/part-r-00016-55b1cd2d-d2c7-43dc-ac2a-da953f82d47b.csv"
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
                allProducts.add(product_id)
                map_customerId_productIdList[customer_id].append(product_id)


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



