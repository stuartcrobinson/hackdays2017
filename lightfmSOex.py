import numpy as np

from lightfm.datasets import fetch_stackexchange

data = fetch_stackexchange('crossvalidated',
                           test_set_fraction=0.1,
                           indicator_features=False,
                           tag_features=True)

train = data['train']
test = data['test']


#Let's examine the data:

print('The dataset has %s users and %s items, '
      'with %s interactions in the test and %s interactions in the training set.'
      % (train.shape[0], train.shape[1], test.getnnz(), train.getnnz()))
      

# Import the model
from lightfm import LightFM

# Set the number of threads; you can increase this
# ify you have more physical cores available.
NUM_THREADS = 2
NUM_COMPONENTS = 30
NUM_EPOCHS = 3
ITEM_ALPHA = 1e-6

# Let's fit a WARP model: these generally have the best performance.
model = LightFM(loss='warp', item_alpha=ITEM_ALPHA, no_components=NUM_COMPONENTS)

# Run 3 epochs and time it.
model = model.fit(train, epochs=NUM_EPOCHS, num_threads=NUM_THREADS)
# %time 

# Import the evaluation routines
from lightfm.evaluation import auc_score

# Compute and print the AUC score
train_auc = auc_score(model, train, num_threads=NUM_THREADS).mean()
print('Collaborative filtering train AUC: %s' % train_auc)

# We pass in the train interactions to exclude them from predictions.
# This is to simulate a recommender system where we do not
# re-recommend things the user has already interacted with in the train
# set.
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


# A hybrid model
# We can do much better by employing LightFM's hybrid model capabilities. 
# The StackExchange data comes with content information in the form of tags users apply to their questions:

item_features = data['item_features']
tag_labels = data['item_feature_labels']

print('There are %s distinct tags, with values like %s.' % (item_features.shape[1], tag_labels[:3].tolist()))
# There are 1246 distinct tags, with values like [u'bayesian', u'prior', u'elicitation'].


# We can use these features (instead of an identity feature matrix like in a pure CF model) 
# to estimate a model which will generalize better to unseen examples: it will simply use its 
# representations of item features to infer representations of previously unseen questions.
# Let's go ahead and fit a model of this type.
# Define a new model instance
model = LightFM(loss='warp',
                item_alpha=ITEM_ALPHA,
                no_components=NUM_COMPONENTS)

# Fit the hybrid model. Note that this time, we pass
# in the item features matrix.
model = model.fit(train,
                item_features=item_features,
                epochs=NUM_EPOCHS,
                num_threads=NUM_THREADS)
# As before, let's sanity check the model on the training set.

# Don't forget the pass in the item features again!
train_auc = auc_score(model,
                      train,
                      item_features=item_features,
                      num_threads=NUM_THREADS).mean()
print('Hybrid training set AUC: %s' % train_auc)
# Hybrid training set AUC: 0.86049
# Note that the training set AUC is lower than in a pure CF model. This is fine: by using a lower-rank
#  item feature matrix, we have effectively regularized the model, giving it less freedom to fit the training data.
# Despite this the model does much better on the test set:


test_auc = auc_score(model,
                    test,
                    train_interactions=train,
                    item_features=item_features,
                    num_threads=NUM_THREADS).mean()
print('Hybrid test set AUC: %s' % test_auc)
# Hybrid test set AUC: 0.703039
# This is as expected: because items in the test set share tags with items in the 
# training set, we can provide better test set recommendations by using the tag representations learned from training.


