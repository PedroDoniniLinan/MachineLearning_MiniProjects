#!/usr/bin/env python
# coding: utf-8

# # Regression

# Prediction of the output of a continuos value (like price or probability)

# In[2]:


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# ## Fecthing data

# In[3]:


dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


# In[3]:


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()


# ##  Cleaning data

# These is the count of unknown values for each feature.

# In[4]:


dataset.isna().sum()


# The simplest solution is to drop these rows.

# In[5]:


dataset = dataset.dropna()


# The Origin feature is categorical not numerical, so we convert it to one-hot.

# In[6]:


origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()
origin.tail()


# ## Preparing train and test set

# In[7]:


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[8]:


sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


# #### Overall statistics

# In[9]:


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats


# #### Split features from labels

# In[10]:


train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# #### Standardization of the data

# Features have very different ranges. It is good practice to standarize them, both training and test set. Afterwards this standardization should be applied to any data supplied to the model.

# In[4]:


def std(x):
  return (x - train_stats['mean']) / train_stats['std'] # std = standard deviation

std_train_data = std(train_dataset)
std_test_data = std(test_dataset)


# ##  Building model

# In[12]:


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


# In[13]:


model = build_model()
model.summary()


# Using a batch of 10 examples, we get a result of expected shape and type

# In[14]:


example_batch = std_train_data[:10]
example_result = model.predict(example_batch)
example_result


# ## Training the model

# In[15]:


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  std_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


# In[16]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[17]:


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)


# This model show degradation of the validation error after around 100 epochs. So we will add an EarlyStopping callback that is checked at end of every epoch and then stop when validation score doesn't improve. 

# In[18]:


model = build_model()

# The patience parameter is the amount of epochs without improvement before training should stop
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(std_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# ## Evaluation

# In[19]:


loss, mae, mse = model.evaluate(std_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# #### Predictions

# The model predicts reasonably well.

# In[20]:


test_predictions = model.predict(std_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# #### Error distribution

# In[21]:


error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# ## Conclusion

# * Mean Squared Error (MSE) is a common loss function used for regression problems (different loss functions are used for classification problems).

# 
# * Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).
# 
# 

# * When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.

# 
# * If there is not much training data, one technique is to prefer a small network with few hidden layers to avoid overfitting.
# 
# 

# * Early stopping is a useful technique to prevent overfitting.
