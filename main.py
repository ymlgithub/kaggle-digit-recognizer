# %%
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# %%
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
data_submission = pd.read_csv('data/sample_submission.csv', index_col=0)
# %%
labels_train = data_train['label'].values
pixels_train = data_train.drop(columns=['label'])
pixels_train = pixels_train.values.reshape((-1, 28, 28))
pixels_test = data_test.values.reshape((-1, 28, 28))
labels_train.shape, pixels_train.shape, pixels_test.shape, data_submission.shape
# %%
plt.figure(figsize=(6, 3))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(pixels_train[i])
    plt.title(labels_train[i])
for i in range(3):
    plt.subplot(2, 3, i+4)
    plt.imshow(pixels_test[i])
# %%


def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, (3, 3)))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# %%
model = get_model()
model.summary()
# %%
ckpt = ModelCheckpoint(
    'tmp/ckpt-{epoch:03d}-acc_{acc:.5f}-val_acc_{val_acc:.5f}.h5', save_best_only=True)
estop = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20)
# %%
model.fit(pixels_train.reshape((-1, 28, 28, 1)),
          keras.utils.to_categorical(labels_train),
          callbacks=[ckpt, estop],
          batch_size=64, epochs=500, validation_split=0.2)


# %%
