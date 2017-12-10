import numpy as np
from lightfm.datasets import fetch_stackexchange
from scipy import sparse
import pickle
from sklearn.externals import joblib
import os.path
import time


start_time = time.time()

# Save the trained model as a pickle string.
# saved_model = pickle.dumps(clf)

# Save the model as a pickle in a file
# joblib.dump(clf, 'filename.pkl') 

# Load the model from the file
# clf_from_joblib = joblib.load('filename.pkl') 

# Use the loaded model to make predictions
# clf_from_joblib.predict(X)


# Check if file exists
# os.path.isfile(fname) 

# https://github.com/lyst/lightfm/blob/master/examples/stackexchange/hybrid_crossvalidated.ipynb
# 
# data = fetch_stackexchange('crossvalidated',
# test_set_fraction=0.1,
# indicator_features=False,
# tag_features=True)
# 
# train = data['train']                    #interactions coo_matrix shape [n_users, n_items]
# test = data['test']
# item_features = data['item_features']    #item_features csr_matrix shape [n_items, n_item_features]


print('hi')

def itemFeaturesData(filename):
    loader = np.load(filename)
    labels = loader['item_feature_labels']
    return labels, sparse.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])
                         
def load_sparse_coo(filename):
    loader = np.load(filename)
    return sparse.coo_matrix((  loader['data'], (loader['row'], loader['col'])),  shape = loader['shape'])

# save_sparse

test = load_sparse_coo("interactionsTest_sparse_coo_matrix.npz")
train = load_sparse_coo("interactionsTrain_sparse_coo_matrix.npz")
item_feature_labels, item_features = itemFeaturesData("itemFeatures_sparse_csr_matrix.npz")

data = {}
data['train'] = train
data['test'] = test
data['item_features'] = item_features
data['item_feature_labels'] = item_feature_labels


#Let's examine the data:
print('The item_features dataset has %s rows and %s columns, with %s nonzero elements.'
% (item_features.shape[0], item_features.shape[1], item_features.getnnz()))

print('\n')

#Let's examine the data:
print(' train.shape[0]: %s \n  train.shape[1]: %s \n  test.shape[0]: %s \n  test.shape[1]: %s \n , '
'with %s interactions in the test and %s interactions in the training set.'
% (train.shape[0], train.shape[1], test.shape[0], test.shape[1], test.getnnz(), train.getnnz()))
# The dataset has 3221 users and 72360 items, with 4307 interactions in the test and 57830 interactions in the training set.
# 
# The training and test set are divided chronologically: the test set contains the 10% of interactions that happened 
# after the 90% in the training set. This means that many of the questions in the test set have no interactions. This is 
# an accurate description of a questions answering system: it is most important to recommend questions that have not yet 
# been answered to the expert users who can answer them.


# quit()

# A pure collaborative filtering model
# 
# This is clearly a cold-start scenario, and so we can expect a traditional collaborative filtering model 
# to do very poorly. Let's check if that's the case:

# Import the model
from lightfm import LightFM

# Set the number of threads; you can increase this
# if you have more physical cores available.
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3  #was 3. 10 is best?  highest test ROC AUC
ITEM_ALPHA = 1e-6


if False and os.path.isfile('filename.pkl'):
    model = joblib.load('filename.pkl')     
else:
    # Let's fit a WARP model: these generally have the best performance.
    model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS)
    
    # Run 3 epochs and time it.
    model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)  # %time <-- screws up regular python run.  jupyter stuff i think.
    # CPU times: user 12.9 s, sys: 8 ms, total: 12.9 s
    # Wall time: 6.52 s
    
    
    # 
# pickledModel = pickle.dumps(model)
#     
# model = pickle.loads(pickledModel)

# joblib.dump(clf, 'filename.pkl') 

    
# As a means of sanity checking, let's calculate the model's AUC on the training set first. 
# If it's reasonably high, we can be sure that the model is not doing anything stupid and is fitting the training data well.

# Import the evaluation routines
from lightfm.evaluation import auc_score

# Compute and print the AUC score
train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)
# Collaborative filtering train AUC: 0.887519
# Fantastic, the model is fitting the training set well. But what about the test set?

# We pass in the train interactions to exclude them from predictions.
# This is to simulate a recommender system where we do not
# re-recommend things the user has already interacted with in the train set.
test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)

# Collaborative filtering test AUC: 0.34728


# This is terrible: we do worse than random! This is not very surprising: as there is no training data
# for the majority of the test questions, the model cannot compute reasonable representations of the test set items.
# The fact that we score them lower than other items (AUC < 0.5) is due to estimated per-item biases,
# which can be confirmed by setting them to zero and re-evaluating the model.

# Set biases to zero
model.item_biases *= 0.0

test_auc = auc_score(model, test, train_interactions=train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering test AUC: %s' % test_auc)
# Collaborative filtering test AUC: 0.496266

# if not os.path.isfile('filename.pkl'):
# saved_model = pickle.dumps(model)

joblib.dump(model, 'filename.pkl') 


print("Elapsed time was %g seconds" % (time.time() - start_time))
quit()

# A hybrid model
# We can do much better by employing LightFM's hybrid model capabilities.
# The StackExchange data comes with content information in the form of tags users apply to their questions:

# item_features = data['item_features']
tag_labels = data['item_feature_labels']

print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))
# There are 1246 distinct tags, with values like [u'bayesian', u'prior', u'elicitation'].


# We can use these features (instead of an identity feature matrix like in a pure CF model)
# to estimate a model which will generalize better to unseen examples: it will simply use its
# representations of item features to infer representations of previously unseen questions.
# Let's go ahead and fit a model of this type.
# Define a new model instance
model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS)

# Fit the hybrid model. Note that this time, we pass
# in the item features matrix.
model = model.fit(train, item_features=item_features, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
# As before, let's sanity check the model on the training set.

# Don't forget the pass in the item features again!
train_auc = auc_score(model, train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Hybrid training set AUC: %s' % train_auc)
# Hybrid training set AUC: 0.86049
# Note that the training set AUC is lower than in a pure CF model. This is fine: by using a lower-rank
# item feature matrix, we have effectively regularized the model, giving it less freedom to fit the training data.
# Despite this the model does much better on the test set:


test_auc = auc_score(model, test, train_interactions=train, item_features=item_features, num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)
# Hybrid test set AUC: 0.703039
# This is as expected: because items in the test set share tags with items in the
# training set, we can provide better test set recommendations by using the tag representations learned from training.




'''
for 3 epochs:
The item_features dataset has 1511 rows and 1411 columns, with 112373 nonzero elements.
 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965601
Collaborative filtering test AUC: 0.921294
Collaborative filtering test AUC: 0.727957

6 epochs:
Collaborative filtering train AUC: 0.989248
Collaborative filtering test AUC: 0.938554
Collaborative filtering test AUC: 0.742754

10: 
Collaborative filtering train AUC: 0.995913
Collaborative filtering test AUC: 0.943351
Collaborative filtering test AUC: 0.749029

15: (overfitting)
Collaborative filtering train AUC: 0.998209
Collaborative filtering test AUC: 0.944166
Collaborative filtering test AUC: 0.747467

12: overfitting
Collaborative filtering train AUC: 0.997085
Collaborative filtering test AUC: 0.94408
Collaborative filtering test AUC: 0.748139

9 under
Collaborative filtering train AUC: 0.995067
Collaborative filtering test AUC: 0.942768
Collaborative filtering test AUC: 0.748672

3 epochs whole thing: hybrid does a little worse / the same  :((((

The item_features dataset has 1511 rows and 1411 columns, with 112373 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.96573
Collaborative filtering test AUC: 0.920734
Collaborative filtering test AUC: 0.725458
There are 1411 distinct tags, with values like ['sacrif', 'oliv', 'hidden'].
Hybrid training set AUC: 0.896416
Hybrid test set AUC: 0.722685

what if we don't stem words? --> gives 431 more tags.  even worse now:


The item_features dataset has 1511 rows and 1842 columns, with 115263 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965421
Collaborative filtering test AUC: 0.920975
Collaborative filtering test AUC: 0.729248
There are 1842 distinct tags, with values like ['sedona', 'weighs', 'paradise'].
Hybrid training set AUC: 0.886427
Hybrid test set AUC: 0.715723

:( :( :( :(

is there a better way to turn descriptions into more tag-like things? 

just titles:  269 "tags" now

WHOA this is much better!!!!

The item_features dataset has 1511 rows and 269 columns, with 9010 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966331
Collaborative filtering test AUC: 0.922175
Collaborative filtering test AUC: 0.727794
There are 269 distinct tags, with values like ['marco', 'satellit', 'group'].
Hybrid training set AUC: 0.976786
Hybrid test set AUC: 0.796275

what if we weight the titles by like 10x?

hmmmm much worse:

hi
The item_features dataset has 1511 rows and 1411 columns, with 112373 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965748
Collaborative filtering test AUC: 0.920462
Collaborative filtering test AUC: 0.729201
There are 1411 distinct tags, with values like ['color', 'paradis', 'indoor'].
Hybrid training set AUC: 0.730215
Hybrid test set AUC: 0.657211

what about just double titles? also bad:

hi
The item_features dataset has 1511 rows and 1411 columns, with 112373 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965337
Collaborative filtering test AUC: 0.918783
Collaborative filtering test AUC: 0.728323
There are 1411 distinct tags, with values like ['cover', 'subject', 'fretwork'].
Hybrid training set AUC: 0.869447
Hybrid test set AUC: 0.707375

what about just using the first 20 words in description?
first 25 words of array of tokens INCLUDING the title - added a few stopwords like "allow" and "exist"

hi
The item_features dataset has 1511 rows and 928 columns, with 32233 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966575
Collaborative filtering test AUC: 0.921165
Collaborative filtering test AUC: 0.724371
There are 928 distinct tags, with values like ['mongolian', 'home', 'key'].
Hybrid training set AUC: 0.966512
Hybrid test set AUC: 0.809244

diff: 85

titles only - with new stopwords:

hi
The item_features dataset has 1511 rows and 269 columns, with 9010 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965518
Collaborative filtering test AUC: 0.921427
Collaborative filtering test AUC: 0.729181
There are 269 distinct tags, with values like ['convers', 'bungalow', 'sunset'].
Hybrid training set AUC: 0.977244
Hybrid test set AUC: 0.788446

diff: 59

first 40 words including title:


hi
The item_features dataset has 1511 rows and 1136 columns, with 49906 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965964
Collaborative filtering test AUC: 0.920908
Collaborative filtering test AUC: 0.728491
There are 1136 distinct tags, with values like ['distress', 'sandi', 'detail'].
Hybrid training set AUC: 0.951866
Hybrid test set AUC: 0.798248

diff: 70

first 15: - best so far


hi
The item_features dataset has 1511 rows and 692 columns, with 19731 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965642
Collaborative filtering test AUC: 0.91961
Collaborative filtering test AUC: 0.727303
There are 692 distinct tags, with values like ['without', 'zen', 'offset'].
Hybrid training set AUC: 0.976739
Hybrid test set AUC: 0.818469

diff: 91

first 10:


hi
The item_features dataset has 1511 rows and 533 columns, with 14065 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966421
Collaborative filtering test AUC: 0.921512
Collaborative filtering test AUC: 0.730994
There are 533 distinct tags, with values like ['zipper', 'cantina', 'fit'].
Hybrid training set AUC: 0.980839
Hybrid test set AUC: 0.819294

diff: 89

full title plus first 10 cleaned description tokens

hi
The item_features dataset has 1511 rows and 686 columns, with 20838 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966118
Collaborative filtering test AUC: 0.920483
Collaborative filtering test AUC: 0.728301
There are 686 distinct tags, with values like ['uir', 'sedona', 'sling'].
Hybrid training set AUC: 0.976456
Hybrid test set AUC: 0.820759

diff: 92

same:


hi
The item_features dataset has 1511 rows and 686 columns, with 20838 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965193
Collaborative filtering test AUC: 0.919984
Collaborative filtering test AUC: 0.727129
There are 686 distinct tags, with values like ['uir', 'sedona', 'sling'].
Hybrid training set AUC: 0.975141
Hybrid test set AUC: 0.818083

diff:  91

title plus first 5 description tokens cleaned


hi
The item_features dataset has 1511 rows and 537 columns, with 15115 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.965465
Collaborative filtering test AUC: 0.919973
Collaborative filtering test AUC: 0.726892
There are 537 distinct tags, with values like ['dove', 'frank', 'coordin'].
Hybrid training set AUC: 0.978916
Hybrid test set AUC: 0.811786

diff:  85

plus 15


hi
The item_features dataset has 1511 rows and 826 columns, with 27051 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966018
Collaborative filtering test AUC: 0.921626
Collaborative filtering test AUC: 0.728864
There are 826 distinct tags, with values like ['satelit', 'gri', 'quit'].
Hybrid training set AUC: 0.970781
Hybrid test set AUC: 0.808196

not as good

title plus first 10 descriptors - w/ some new stopwords - 'qualiti' 'use' 'live' etc


hi
The item_features dataset has 1511 rows and 687 columns, with 20793 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.96512
Collaborative filtering test AUC: 0.919395
Collaborative filtering test AUC: 0.728468
There are 687 distinct tags, with values like ['frank', 'multi', 'bar'].
Hybrid training set AUC: 0.976226
Hybrid test set AUC: 0.817038

diff: 89

title plus 13 cleaned


hi
The item_features dataset has 1511 rows and 776 columns, with 24532 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.966221
Collaborative filtering test AUC: 0.921315
Collaborative filtering test AUC: 0.72889
There are 776 distinct tags, with values like ['bliss', 'weatherproof', 'distress'].
Hybrid training set AUC: 0.97379
Hybrid test set AUC: 0.820592

diff:  92

again: - title plus 13 cleaned


hi
The item_features dataset has 1511 rows and 776 columns, with 24532 nonzero elements.


 train.shape[0]: 42937
  train.shape[1]: 1330
  test.shape[0]: 42937
  test.shape[1]: 1330
 , with 23800 interactions in the test and 74704 interactions in the training set.
Collaborative filtering train AUC: 0.964994
Collaborative filtering test AUC: 0.919816
Collaborative filtering test AUC: 0.724909
There are 776 distinct tags, with values like ['bliss', 'weatherproof', 'distress'].
Hybrid training set AUC: 0.973435
Hybrid test set AUC: 0.821669

diff:  97

okay.... now what.  how to get suggestion from inputs????

TODO - build test matrix from a single user and a set of product ids.

TODO - figure out how to save and load model

'''