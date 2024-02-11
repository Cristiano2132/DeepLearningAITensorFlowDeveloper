import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

print(tf.__version__)

start_time = time.time()
# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# # Visualize the image
# plt.imshow(training_images[index])
# plt.show()
# plt.close()
# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Build the classification model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Sequential: That defines a sequence of layers in the neural network.

# Flatten: Remember earlier where our images were a 28x28 pixel matrix when you printed them out? Flatten just takes that square and turns it into a 1-dimensional array.

# Dense: Adds a layer of neurons

# Each layer of neurons need an activation function to tell them what to do. There are a lot of options, but just use these for now:

# ReLU effectively means:

# if x > 0: 
#   return x

# else: 
#   return 0
# In other words, it only passes values greater than 0 to the next layer in the network.

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
evaluation = model.evaluate(test_images, test_labels)
print(f'LOSS: {evaluation[0]}')
print(f'ACCURACY: {evaluation[1]}')

classifications = model.predict(test_images)
end_time = time.time()
print(f'TIME ELAPSED: {end_time - start_time}')

start_time = time.time()
fmnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels) ,  (test_images, test_labels) = fmnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), # Try experimenting with this layer
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

evaluation = model.evaluate(test_images, test_labels)
print(evaluation)
print(f'LOSS: {evaluation[0]}')
print(f'ACCURACY: {evaluation[1]}')

classifications = model.predict(test_images)
end_time = time.time()
end_time = time.time()
print(f'TIME ELAPSED: {end_time - start_time}')


start_time = time.time()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels) ,  (test_images, test_labels) = fmnist.load_data()

training_images=training_images/255.0
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])

evaluation = model.evaluate(test_images, test_labels)
print(f'LOSS: {evaluation[0]}')
print(f'ACCURACY: {evaluation[1]}')
end_time = time.time()
print(f'TIME ELAPSED: {end_time - start_time}')