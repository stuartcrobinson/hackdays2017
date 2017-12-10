from __future__ import print_function
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy import sparse
 
import glob
import csv
import sys  
import io
import html
import string
import numpy
# import StringIO
# https://stackoverflow.com/questions/26369051/python-read-from-file-and-remove-non-ascii-characters
# TODO
# 
# reload(sys)  
# sys.setdefaultencoding('utf8')

#interactions coo_matrix shape [n_users, n_items]
#item_features csr_matrix shape [n_items, n_item_features]
# item_feature_labebs

'''


for custom NN .... what to generate

'''

stemmer = PorterStemmer()

stopwords = ['qualiti', 'live', 'use', 'exist', 'allow', 'ad', 'start', 'make', 'way', 'rst', 'll', 're', 'd', 've', 's', 't', 'br', 'li', 'nbsp', 'p', 'span', 'div', 'ul', 'ol', 'includes','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

def get_words(doc):
    # replace punctuation with space
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    doc = doc.translate(replace_punctuation)
    # split into tokens by white space
    tokens = doc.split()    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if (word.isalpha() and word not in stopwords)]
    # make lower case
    tokens = [stemmer.stem(word.lower()) for word in tokens]
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

# 
# def removeStopwords(words, stopwords):
#     resultwords  = [word for word in words if word not in stopwords]
#     return resultwords
# 
# def clean(text):
#     return removeStopwords(get_words(text), stopwords)

#product_id,active,title,description


path = "fordocker/products_production_siteId_=35569/*.csv"
d = {}
for fname in glob.glob(path):
    print(fname)
    nonascii = bytearray(range(0x80, 0x100))
#     with open(fname,'rb') as infile, open('fordocker/d_parsed.csv','wb') as outfile:
    with open(fname,'rb') as infile:
        next(infile)
        for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
            line = line.translate(None, nonascii)
            line = str(line,'utf-8')
            for row in csv.reader([line]):
#                 print(row[0])
                k,v  = row[0], get_words(row[2]) + get_words(row[3])[:13]  #    #title plus description    #k,v  = row[0], clean(row[2] + " " + row[3])[:-1]    <-- all but last word
#                 v = row[3]
                d[k] = v

                
print(len(d))

shared = {}

tags = {}

for key in d:
    print(key)
    shared = set(d[key])
    tags = set(d[key])
#     print('original shared:')
#     print(shared)
    break

print('')

for key in d:
    print(key)# 
    print((d[key]))
    shared = shared.intersection(d[key])
    tags = tags.union(d[key])
#     print('new shared:')
#     print(shared)
#     print('new union:')
#     print(tags)
#     print(html.unescape(d[key]))

print('shared tokens (could be none)')
print(shared)
print('all tokens')
print(tags)
print('')

#done making item features!!!!!!   what next?  collect all tags into a single list/vector.  now, made maps!  seeeeee

tags = list(tags)

map_index_tag = dict(enumerate(tags))
map_tag_index = {x:i for i,x in enumerate(tags)}
# print(b)

# okay next .... build into a matrix . then convert to csr or coo idk.  
# rows: items
# cols: features vector

# a = numpy.zeros(shape=(5,2))
a = numpy.zeros(shape=(len(d),len(tags)))

orderedProductIds = []

for i, key in enumerate(d):
    orderedProductIds.append(key)
    for tag in d[key]:
        tag_index = map_tag_index[tag]
        a[i][tag_index] += 1



r = 0
print(orderedProductIds[r])        
print()


for i in range(0, len(a[r])):
    if a[r][i] > 0:
        print(map_index_tag[i], ": ", a[r][i])


# next build interaction matrix
# convert a to csr_matrix
# https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix

itemFeatures_sparse_csr_matrix = sparse.csr_matrix(a)

# item_feature_labels
# def save_sparse_csr(filename,array):
#     numpy.savez(filename,data = array.data ,indices=array.indices,
#              indptr =array.indptr, shape=array.shape )
# 
# def load_sparse_csr(filename):
#     loader = numpy.load(filename)
#     return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
#                          shape = loader['shape'])
# 
# 
# save_sparse_csr('itemFeatures_sparse_csr_matrix',itemFeatures_sparse_csr_matrix)

filename = 'fordocker/itemFeatures_sparse_csr_matrix'
array = itemFeatures_sparse_csr_matrix

numpy.savez(filename, item_feature_labels=numpy.array(tags), data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

# numpy.savez('itemFeatures_sparse_csr_matrix', matrix=itemFeatures_sparse_csr_matrix)


#now create coo matrix from interactions.  need browse.  lets make new python file