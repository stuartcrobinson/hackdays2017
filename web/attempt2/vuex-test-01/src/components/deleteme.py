# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences

def asdf():

  from keras.models import Sequential
  from keras.layers import Dense

  model = Sequential()

  model.add(Dense(2 * numProducts,
                  input_dim=numProducts,
                  activation='relu'))

  model.add(Dense(numProducts,
                  activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])





model.fit(X, y, epochs=3, batch_size=2000)


