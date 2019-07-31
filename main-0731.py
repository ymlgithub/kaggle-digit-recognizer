# %%
import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
# %%
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
data_submission = pd.read_csv('data/sample_submission.csv', index_col=0)
# %%
labels_train = data_train['label'].values
pixels_train = data_train.drop(columns=['label'])
pixels_train = pixels_train.values.reshape((-1, 28, 28)).astype('float')
pixels_test = data_test.values.reshape((-1, 28, 28)).astype('float')
# 归一化
pixels_train /= 255.
pixels_test /= 255.

labels_train.shape, pixels_train.shape, pixels_test.shape, data_submission.shape
# %%
# plt.figure(figsize=(6, 3))
# for i in range(3):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(pixels_train[i])
#     plt.title(labels_train[i])
# for i in range(3):
#     plt.subplot(2, 3, i+4)
#     plt.imshow(pixels_test[i])
# %%


def get_model():
    def res_block(x):
        y = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
        return concatenate([x, y])
    x_input = Input(shape=(28, 28, 1))

    x = Conv2D(128, (5, 5), activation='relu')(x_input)
    x = MaxPool2D((2, 2))(x)

    # 3:0.99342
    for i in range(5):
        x = res_block(x)
        x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_model()
model.summary()
# %%
ckpt = ModelCheckpoint(
    'tmp/ckpt-'+time.strftime('%Y-%m-%d_%H_%M')+'-Epoch_{epoch:03d}-acc_{acc:.5f}-val_acc_{val_acc:.5f}.h5', save_best_only=True, monitor='val_acc')
estop = EarlyStopping(monitor='val_acc', min_delta=1e-7,
                      verbose=1, patience=20)
# %%
training = True
# training = False
# %%
if training:
    model.fit(pixels_train.reshape((-1, 28, 28, 1)),
              keras.utils.to_categorical(labels_train),
              callbacks=[ckpt, estop],
              batch_size=256, epochs=500, validation_split=0.2)
else:
    model = load_model(
        'tmp/ckpt-2019-07-31_12_09-Epoch_113-acc_0.99926-val_acc_0.99417.h5')
# %%
preds = model.predict(pixels_test.reshape((-1, 28, 28, 1)))


# %%
labels_pred = np.argmax(preds, axis=1)
data_submission['Label'] = labels_pred
data_submission.to_csv('submission.csv')

# %%
plt.figure(figsize=(12, 5))
for i in range(6):
    plt.subplot(2, 6, i+1)
    plt.imshow(pixels_train[i])
    plt.title(labels_train[i])
for i in range(6):
    plt.subplot(2, 6, i+7)
    plt.imshow(pixels_test[i])
    plt.title(labels_pred[i])

plt.show()
# %%
