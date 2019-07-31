# kaggle-digit-recognizer

[digit-recognizer](https://www.kaggle.com/c/digit-recognizer)

# 经验

归一化和标准化效果差不多

# 结果
```
# 0.97542   ckpt-056-acc_0.99997-val_acc_0.97560
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# 0.97800   ckpt-041-acc_0.99777-val_acc_0.98214
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3)))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# 0.98485   ckpt-065-acc_0.99857-val_acc_0.98619
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# 0.98585  ckpt-084-acc_0.99976-val_acc_0.98690
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# ckpt-131-acc_0.99098-val_acc_0.99226
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# ckpt-092-acc_0.99000-val_acc_0.99155
model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# 0.99042   ckpt-118-acc_0.99360-val_acc_0.99274
model = Sequential()
model.add(Conv2D(128, (5, 5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

```
# ckpt-2019-07-31_11_44-Epoch_088-acc_0.99473-val_acc_0.99369
model = Sequential()
model.add(Conv2D(128, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
# model.add(Conv2D(128, (3, 3)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```