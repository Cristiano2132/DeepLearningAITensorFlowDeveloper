# The Data: Fashion MNIST One of the foundational datasets for learning and benchmarking algorithms 
# is the Modified National Institute of Standards and Technology (MNIST) database, by Yann LeCun, 
# Corinna Cortes, and Christopher Burges. This dataset is comprised of images of 70,000 handwritten 
# digits from 0 to 9. The images are 28 × 28 grayscale. Fashion MNIST is designed to be a drop-in 
# replacement for MNIST that has the same number of records, the same image dimensions, and the same 
# number of classes—so, instead of images of the digits 0 through 9, Fashion MNIST contains images of 
# 10 different types of clothing.

# It has a nice variety of clothing, including shirts, trousers, dresses, and lots of types of shoes. 
# As you may notice, it’s monochrome, so each picture consists of a certain number of pixels with 
# values between 0 and 255. This makes the dataset simpler to manage.

# Each of our images is a set of 784 values (28 × 28) between 0 and 255. They can be our X. 
# We know that we have 10 different types of images in our dataset, so let’s consider them to be our Y.
# Now we want to learn what the function looks like where Y is a function of X.

import tensorflow as tf
# load the data
data = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# normalize the data
training_images  = training_images / 255.0
test_images = test_images / 255.0

# build the model
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)), # input layer specification
            tf.keras.layers.Dense(128, activation=tf.nn.relu), # hidden layer (middle layer) specification
            tf.keras.layers.Dense(10, activation=tf.nn.softmax) # output layer specification
        ])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

eval = model.evaluate(test_images, test_labels)
print(f'Accuracy: {eval[1]}')
classifications = model.predict(test_images)

label_map = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

for prob, lab in zip(classifications[0], test_labels):
    print(f'{label_map.get(lab)}: {round(prob*100,3)}%')