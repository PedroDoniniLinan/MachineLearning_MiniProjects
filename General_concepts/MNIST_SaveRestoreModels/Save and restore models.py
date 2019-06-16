#!/usr/bin/env python
# coding: utf-8

# # Save and restore models

# Model progress can be saved during — and after — training. It is good practice to share:
# * code to create a model
# * the trained weights and parameters
# 
# How to save a model depends on the API used

# In[1]:


from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__


# #### Loading data

# Here the simple and well-known dataset MNIST is used, since the goal is to explore the saving and restoring mechanisms.

# In[2]:


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0 # -1 infers the shape => 1000 in this case (nb of examples)
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# #### Building a simple model

# This model uses sparse categorical crosentropy, because the target labels are integers. If we used categorical crossentropy we would need to encode them with one-hot.

# In[17]:


# Returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])
  
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  return model


# Create a basic model instance
model = create_model()
model.summary()


# ## Save checkpoints during training

# The primary use case is to automatically save checkpoints during and at the end of training. This way we can use a trained model without having to retrain it, or pick-up training where you left of—in case the training process was interrupted.

# In[18]:


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels,  epochs = 10, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# In[5]:


model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# By loading from the checkpoint, we can see it is indeed the same model trained before.

# In[6]:


model.load_weights(checkpoint_path)

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[7]:


# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)


# In[8]:


get_ipython().system(' dir {checkpoint_dir}')


# In[9]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[10]:


model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# ## Manually save weigths

# In[11]:


# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# ##  Save the entire model

# The entire model can be saved to a file that contains:
# * the weight values
# * the model's configuration
# * the optimizer's configuration (depends on set up). 
# 
# This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.

# Saving a fully-functional model is very useful—you can load them in TensorFlow.js (HDF5, Saved Model) and then train and run them in web browsers, or convert them to run on mobile devices using TensorFlow Lite (HDF5, Saved Model).

# ####  HDF5 file

# In[12]:


model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')


# In[13]:


# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()


# In[14]:


loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# Currently, it is not able to save TensorFlow optimizers (from tf.train).

# #### saved_model

# In[22]:


model = create_model()

model.fit(train_images, train_labels, epochs=5)

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[23]:


saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")


# In[24]:


new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()


# In[25]:


# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

