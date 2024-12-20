import tensorflow as tf
from keras.datasets import mnist
from keras import layers
from keras.models import Sequential


(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# Preprocessing the input data
# Reshape to be samples*pixels*width*height
train_img = train_img.reshape(-1, 28, 28, 1)
test_img = test_img.reshape(-1, 28, 28, 1)

train_img = train_img / 255.0
test_img = test_img / 255.0

# One hot Cpde
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Defining the model architecture
model = Sequential()

model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_img, train_labels, validation_data=(test_img, test_labels), epochs=10)

val_loss, val_acc = model.evaluate(test_img, test_labels)
print(val_loss, val_acc)
model.save('mnist.h5')
