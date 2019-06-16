#!/usr/bin/env python
# coding: utf-8

# # Text classification - IMDB reviews

# This notebook classifies movie reviews as positive or negative using the text of the review. This is an example of binary—or two-class—classification, an important and widely applicable kind of machine learning problem.

# In[20]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# I will use the IMDB dataset that contains the text of 50,000 movie reviews from the Internet Movie Database. These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are balanced, meaning they contain an equal number of positive and negative reviews.
# 
# The IMDB dataset comes packaged with TensorFlow. It has already been preprocessed such that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary.
# 
# The argument num_words=10000 keeps the top 10,000 most frequently occurring words in the training data. The rare words are discarded to keep the size of the data manageable.

# In[2]:


imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# ### Exploring the data

# The dataset comes preprocessed: each example is an array of integers representing the words of the movie review. Each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.
# 
# The text of reviews have been converted to integers, where each integer represents a specific word in a dictionary. Movie reviews may be different lengths and this will be solved later since the network needs inputs of same length.

# In[6]:


print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])


# ### Preparing the data

# Here is a helper function to query a dictionary object that contains the integer to string mapping which can be useful.

# In[7]:


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])


# The reviews—the arrays of integers—must be converted to tensors before fed into the neural network. This conversion can be done in a couple of ways:
# 
# 1. Convert the arrays into vectors of 0s and 1s indicating word occurrence, similar to a one-hot encoding. For example, the sequence [3, 5] would become a 10,000-dimensional vector that is all zeros except for indices 3 and 5, which are ones. Then, make this the first layer in our network—a Dense layer—that can handle floating point vector data. This approach is memory intensive, though, requiring a num_words * num_reviews size matrix.
# 
# 
# 2. Alternatively, we can pad the arrays so they all have the same length, then create an integer tensor of shape max_length * num_reviews. We can use an embedding layer capable of handling this shape as the first layer in our network.
# 
# I will use the second approach since it is more memory efficient.

# In[9]:


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[10]:


len(train_data[0]), len(train_data[1])


# In[12]:


print(train_data[0])


# ### Building the model

# The layers are stacked sequentially to build the classifier:
# 
# 1. The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
# 
# 
# 2. Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
# 
# 
# 3. This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
# 
# 
# 4. The last layer is densely connected with a single output node. Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability, or confidence level.

# The choosen loss function is the binary_crossentropy which is good for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions, specifically for binary classifications in comparison to the categorical_crossentropy. The optimizer is the classical and one of the best, the adam optimizer.

# In[15]:


# input shape is the vocabulary count used for the movie reviews (10,000 words)

def build_model():
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    
    return model

model = build_model()
model.summary()


# ### Training the model

# Train the model for 40 epochs in mini-batches of 512 samples. This is 40 iterations over all samples in the train_data and train_labels tensors with exception to the validation set data. While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set:

# In[16]:


model = build_model()

history = model.fit(train_data,
                    train_labels,
                    epochs=40,
                    batch_size=512,
                    validation_split=0.4,
                    verbose=1)


# ### Evaluating the model

# This fairly naive approach achieves an accuracy of about 87%. With more advanced approaches, the model should get closer to 95%.

# In[18]:


results = model.evaluate(test_data, test_labels)


# In[21]:


def plot_history(history, metric='loss', label='Loss', scaling=1):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.plot(hist['epoch'], hist['val_' + metric]*scaling,
           label='Validation ' + label)
    plt.plot(hist['epoch'], hist[metric]*scaling,
           label='Train ' + label)
    plt.legend()

plot_history(history)
plot_history(history, 'acc', 'Accuracy (%)', 100)


# ### Re-training an re-evaluating the model

# The validation loss and accuracy seem to peak after about twenty epochs. This is an example of overfitting: the model performs better on the training data than it does on data it has never seen before. After this point, the model over-optimizes and learns representations specific to the training data that do not generalize to test data. So an early stop callback is desirable here.

# In[30]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

model = build_model()

history = model.fit(train_data,
                    train_labels,
                    epochs=100,
                    batch_size=512,
                    validation_split=0.4,
                    callbacks=[early_stop],
                    verbose=2)


# We can verify that the early stopping saves training time and gives a sligthly better model.

# In[31]:


results = model.evaluate(test_data, test_labels)


# In[32]:


plot_history(history)
plot_history(history, 'acc', 'Accuracy (%)', 100)

