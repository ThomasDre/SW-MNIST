import os

import tensorflow-cpu as tf
from keras.datasets import mnist
from matplotlib import pyplot

(train_x, train_y), (test_x, test_y) = mnist.load_data()

print(train_x.shape)


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x=train_x, y=train_y, epochs=15, batch_size=32, validation_split=0.1)

model.save('model')
