import os
from dataset import prepare_image, preprocess_image
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

X_train, y_train = prepare_image('./data/train_list.txt')
train_images = []
for i in tqdm(X_train):
    train_image = preprocess_image(i)
    train_images.append(train_image)

from tensorflow import keras

train_images = np.array(train_images)
print(train_images.shape)
y_train = keras.utils.to_categorical(y_train, 12)


class Vgg_Block(keras.layers.Layer):
    def __init__(self, units, filters, **kwargs):
        super().__init__(**kwargs)
        self.main_layer = []
        for i in range(units):
            self.main_layer.append(keras.layers.Conv2D(filters=filters, kernel_size=(3, 2),
                                                       padding="same", strides=(1, 1),
                                                       activation="relu"))
        self.main_layer.append(keras.layers.MaxPool2D(pool_size=(2, 2)))

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layer:
            Z = layer(Z)
        return Z


model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(224, 224, 3)))
model.add(Vgg_Block(2, 64))
model.add(Vgg_Block(2, 128))
model.add(Vgg_Block(4, 256))
model.add(Vgg_Block(4, 512))
model.add(Vgg_Block(4, 512))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(4096, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(12, activation="softmax"))

# keras.utils.plot_model(model=model, to_file='AlexNet.png', show_shapes=True)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
history = model.fit(train_images, y_train, epochs=20, batch_size=16, validation_split=0.2)

model.save('the_AlexNet_model.h5')


def show_training_history(train_history, train, val):
    plt.plot(train_history[train], linestyle='-', color='b')
    plt.plot(train_history[val], linestyle='--', color='r')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('train', fontsize=12)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


show_training_history(history.history, 'loss', 'val_loss')
show_training_history(history.history, 'acc', 'val_acc')