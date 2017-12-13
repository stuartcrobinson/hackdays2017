# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences

def asdf():

  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()

  model.add(Dense(2 * numInputNodes,
                  input_dim=numInputNodes,
                  activation='relu'))

  model.add(Dense(len(map_productId_index),
                  activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])








