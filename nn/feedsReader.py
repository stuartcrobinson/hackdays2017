import numpy
import csv
import glob
import os

ids = []
urls=[]

print("hi")

root = os.path.expanduser('~/ml/')

# 
# with open('feed_35569.txt','r') as f:
#     next(f) # skip headings
#     reader=csv.reader(f, delimiter='\t')
#     for product_id, product_url in reader:
#         ids.append(product_id)
#         urls.append(product_url)
# print(ids)

map_productId_imgUrl = {}
map_productId_parentProductId ={}
map_productId_title = {}

siteId = '38178'


if siteId=='35569':
    iId = 0
    iUrl = 7
    iParent = 15
elif siteId=='38178':
    iId = 0
    iUrl = 12
    iParent = 17
    iTitle = 1


path = 'feed_'+siteId+'.txt'
nonascii = bytearray(range(0x80, 0x100))
for fname in glob.glob(path):
    print(fname)
    with open(fname, mode='rb') as infile:
        next(infile)
        for line in infile: # b'\n'-separated lines (Linux, OSX, Windows)
            line = line.translate(None, nonascii)
            line = str(line,'utf-8')
            reader = csv.reader([line], delimiter='\t')
            for row in reader:
                id = row[iId]
                url = row[iUrl]
                title = row[iTitle]
                parentId = row[iParent]
                if len(parentId)==0 or parentId == ' ':
                    parentId = id
                map_productId_imgUrl[id] = url
                map_productId_parentProductId[id] = parentId
                map_productId_title[id] = title


print("len(map_productId_imgUrl):", len(map_productId_imgUrl))
print("len(map_productId_parentProductId):", len(map_productId_parentProductId))
print("len(map_productId_title):", len(map_productId_title))
# print(map_productId_parentProductId['OP-ALCLB5-AST-NVY-K'])

numpy.save(root + 'dataMedium/map_productId_imgUrl_'+siteId+'.npy', map_productId_imgUrl)
numpy.save(root + 'dataMedium/map_productId_title_'+siteId+'.npy', map_productId_title)
numpy.save(root + 'dataMedium/map_productId_parentProductId_'+siteId+'.npy', map_productId_parentProductId)

