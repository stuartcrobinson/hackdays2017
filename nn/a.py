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


stemmer = PorterStemmer()

def get_words(doc):
    # replace punctuation with space
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    doc = doc.translate(replace_punctuation)
    # split into tokens by white space
    tokens = doc.split()    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [stemmer.stem(word.lower()) for word in tokens]
    return tokens

def removeStopwords(words, stopwords):
    resultwords  = [word for word in words if word not in stopwords]
    return resultwords

stopwords = ['rst', 'll', 're', 'd', 've', 's', 't', 'br', 'li', 'nbsp', 'p', 'span', 'div', 'ul', 'ol', 'includes','a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']

def clean(text):
    return removeStopwords(get_words(text), stopwords)

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
                print(row[0])
                k,v  = row[0], clean(row[2] + " " + row[3])[:-1]
#                 v = row[3]
                d[k] = v

                
print(len(d))

shared = {}

tags = {}

for key in d:
    print(key)
    shared = set(d[key])
    tags = set(d[key])
    print('original shared:')
    print(shared)
    break

print('')

for key in d:
    print(key)
    print((d[key]))
    shared = shared.intersection(d[key])
    tags = tags.union(d[key])
    print('new shared:')
    print(shared)
    print('new union:')
    print(tags)
#     print(html.unescape(d[key]))

print('shared tokens (could be none)')
print(shared)
print('all tokens')
print(tags)
print('')

#done making item features!!!!!!   what next?  collect all tags into a single list/vector.  now, made maps!  seeeeee

map_index_tag = dict(enumerate(tags))
map_tag_index = {x:i for i,x in enumerate(tags)}
print(b)

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





quit()

# 
# query = 'What well that is great hello oh yeah cool '
# querywords = query.split()
# 
# removeStopwords(querywords, stopwords)
# # 
# 
#     
# x = "asdf;a,a .....,.,.,.<li>asdf<br><br><br>.,.a,a. a,f.sd,f, s,.adf"
# 
# # words = clean_doc(x)
# words = removeStopwords(get_words(x),stopwords)
# 
# x
# words    
    
# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', ' ', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens
    
    
x = "asdf;a,a.a,a.a,f.sd,f,s,.adf"
clean_doc(x)


    
    
    # uh so .... 
# 
#             
# path = "fordocker/products_production_siteId_=35569/*.csv"
# for fname in glob.glob(path):
#     print(fname)
#     nonascii = bytearray(range(0x80, 0x100))
# #     with open(fname,'rb') as infile, open('fordocker/d_parsed.csv','wb') as outfile:
#     with open(fname,'rb') as infile, open('fordocker/d_parsed2.csv','w') as outfile:
#         for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
#             line2 = line.translate(None, nonascii)
#             line2 = str(line2,'utf-8')
# #             print(line2)
#             outfile.write(line2)
# #             row = csv.reader(StringIO.StringIO(line2))
#             for row in csv.reader([line2]):
#                 print(row)
# #             print(row)
#             # k = row[0]
# #             v = row[3]
# #             d[k] = v
# # 
# # print(d)            

    
    # successfully reads the whole file.  gives error during output
#     with io.open(fname,'r',encoding='utf-8',errors='ignore') as infile:
#         for line in infile:
#             print(len(line))
#     data = ""
#     with open(fname, 'r') as myfile:
#         data=myfile.read()
#     print(data)
#     with open(fname, encoding='utf-8') as f:
#         while True:
#             c = f.read(1)
#             if not c:
#                 print("End of file")
#                 break
#             print("Read a character:", c)
#      
   #  
#     with open(fname, "r") as ins:
#         array = []
#         for line in ins:
# #             array.append(line)
#             print(line)
#     
#     with open(fname, encoding='ascii') as f:
#         content = f.readlines()
#         print(content)
#  #    
#     reader = csv.reader(open(fname, 'r'))
#     print(reader)
#     for row in reader:
#         print(row)
#         k = row[0]
#         v = row[3]
#         d[k] = v
#     
# #    read file into dictionary     0 to 3:  product_id,active,title,description
#     with open(fname, mode='r') as infile:
#         reader = csv.reader(infile)
#         with open('coors_new.csv', mode='w') as outfile:
#             writer = csv.writer(outfile)
#             mydict = {rows[0]:rows[1] for rows in csv.reader(infile)}

# print(len(d))





# path = "fordocker/browse_production_siteId_=35569/*.csv"
# for fname in glob.glob(path):
#     print(fname)

print('hi bye')
print('what okay')
quit()
