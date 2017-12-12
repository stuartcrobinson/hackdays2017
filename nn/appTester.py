import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def baseline_model(map_productId_index):
    model = Sequential()
    model.add(Dense(len(map_productId_index), input_dim=len(map_productId_index), activation='relu'))
    model.add(Dense(len(map_productId_index))) #makes loss very slightly less.  by 0.01 at epoch 11
    model.add(Dense(len(map_productId_index), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def getRecs(filename, model, productIdsNew):
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    Xnew = numpy.zeros(shape=(1, len(map_productId_index))) #map_productId_index
    for productId in productIdsNew:
        if productId not in map_productId_index:
            return "invalid product: " + productId
        productIndex = map_productId_index[productId]
        Xnew[0][productIndex] = 1
    prediction = model.predict(Xnew, verbose=1)
    pmax1 = -1
    imax1 = 0
    pmax2 = -1
    imax2 = 0
    pmax3 = -1
    imax3 = 0
    for i, p in enumerate(prediction[0]):
        if p > pmax1 and map_index_productId[i] not in productIdsNew:
            pmax3 = pmax2
            pmax2 = pmax1
            pmax1 = p
            imax3 = imax2
            imax2 = imax1
            imax1 = i
        else:
            pass
    prodId1 = map_index_productId[imax1]
    prodId2 = map_index_productId[imax2]
    prodId3 = map_index_productId[imax3]
    recs = [prodId1, prodId2, prodId3]
    return recs

map_productId_index = numpy.load('dataMedium/map_productId_index.npy').item()

map_index_productId = {}
for k in map_productId_index:
    map_index_productId[map_productId_index[k]] = k
print(len(map_productId_index))

filename = "weights/weightsMedium-improvement-09-2.9488.hdf5" #weightsMedium-improvement-50-1.4660.hdf5"
# filename = "weightsMedium-improvement-09-2.9488.hdf5" #weightsMedium-improvement-50-1.4660.hdf5"
model = baseline_model(map_productId_index)
productIdsNew = ['OP-PEBS2-CNS-SPA-K','OP-PEBST5-BLS-K','IP-PEBS2-PORIII-GRY'] #stools

print(getRecs(filename, model, productIdsNew))
