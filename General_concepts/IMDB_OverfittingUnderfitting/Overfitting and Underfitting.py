#!/usr/bin/env python
# coding: utf-8

# # Overfitting and underfitting

# #### Overfit

# When the model adapts to the particularities of the training set. We want to generalize it for better performance in unknown data.
# 
# It can happen when:
# - we train the model for too long (the validation accuracy hits a peak and starts decreasing)
# - the model has too much freedom (too many layers or hidden units)
# 
# Solutions:
# 1. Train on large datasets
# 2. Use regularization
# 3. Find the best training duration

# #### Underfit

# When there is still room for improvement on test data.
# 
# It can happen when:
# - the model is not powerful enough
# - it is overregularized
# - was not trained enough
# 

# In[1]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## Preprocessing

# We will use the IMDB review dataset with multi-hot encoding (\[3, 5\] => \[000101000...00\])  instead of embedding. The model will quickly overfit.

# In[2]:


NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)


# Since the words are sorted byt frequence, we expect more 1's near index 0

# In[3]:


plt.plot(train_data[0])


# ## About overfitting

# The simplest way to prevent overfitting is to reduce the size (**capacity** = number of learnable parameters = number of layers and units) so that it will focus on more important patterns with more predictive power.
# 
# To find the best architecture, the best is to start with just a few layers and units and then increase them until validation loss stops improving.

# ####  Baseline model

# In[4]:


baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works. 
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()


# In[5]:


baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)


# #### Smaller model

# In[6]:


smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()


# In[7]:


smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)


# #### Bigger model

# In[8]:


bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()


# In[9]:


bigger_history = bigger_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)


# In[17]:


def plot_history(histories, key='binary_crossentropy', y_min=0):
    plt.figure(figsize=(8,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.ylim([y_min, 1])


plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history)])


# The bigger network starts to overfit after one epoch and much more severly. The smaller starts to overfit after the baseline and more slowly. So more capacity, means low training loss and high suceptibility to overfit (large difference between training and validation loss).

# ## Strategies

# ### Weight regularization

# The simpler models are usually the best against overfitting. So if it has less parameters or if they have less entropy it is simpler. Thus forcing the weights to take small values, makes the distribution more "regular".
# 
# This is done by adding to the loss function a cost associated to large weights. There are two possibilities:
# - L1 regularization: cost added is proportional to the weights (L1 norms)
# - L2 regularization (weight decay): cost added is proportional to the square of the weights (L2 norms)

# In[11]:


l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)


# l2(0.001) means that every weight in the weight matrix will add 0.001w^2 to the total loss. Because of this the loss during training is much higher than during testing.

# In[18]:


plot_history([('baseline', baseline_history),
              ('l2', l2_model_history)])


# ### Dropout

# One of the most effective regularization techniques. It consistis on dropping out (i.e. set to zero) some output features of the layer during training randomly.
# 
# Ex.: A layer output \[0.2, 0.5, 1.3, 0.8, 1.1\] =>  \[0, 0.5, 1.3, 0, 1.1\] (random case)
# 
# The "dropout rate" is the fraction of features to  be set to zero (usually between \[0.2, 0.5\]). This is not done during testing, instead outputs are scaled down by the dropout rate to compensate.

# In[13]:


dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=2)


# In[19]:


plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history)])


# ### Combining both strategies

# In[15]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

all_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

all_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

all_model_history = all_model.fit(train_data, train_labels,
                                  epochs=20,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  verbose=1)
                                  #callbacks=[early_stop])


# In[21]:


plot_history([('baseline', baseline_history),
              ('dropout', dpt_model_history),
              ('l2', l2_model_history),
              ('all', all_model_history)])


# ##  Conclusion

# Adding dropout is a clear improvement over the baseline model.
# 
# To recap: here the most common ways to prevent overfitting in neural networks:
# 
# - Get more training data.
# - Reduce the capacity of the network.
# - Add weight regularization.
# - Add dropout.
# - And two important approaches not covered in this notebook are data-augmentation and batch normalization.
