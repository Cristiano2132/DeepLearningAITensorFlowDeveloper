import os
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def plot_images(train_horse_dir, train_human_dir, train_horse_names, train_human_names):
      # Parameters for our graph; we'll output images in a 4x4 configuration
      nrows = 4
      ncols = 4
      # Index for iterating over images
      pic_index = 0

      # Set up matplotlib fig, and size it to fit 4x4 pics
      fig = plt.gcf()
      fig.set_size_inches(ncols * 4, nrows * 4)

      pic_index += 8
      next_horse_pix = [os.path.join(train_horse_dir, fname)
                      for fname in train_horse_names[pic_index-8:pic_index]]
      next_human_pix = [os.path.join(train_human_dir, fname)
                      for fname in train_human_names[pic_index-8:pic_index]]

      for i, img_path in enumerate(next_horse_pix+next_human_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

      plt.show()

if __name__ == '__main__':
    
    # Download data from kaggle https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset?resource=download&select=horse-or-human

    # Exploratory
    # Directory with our training horse pictures
    train_horse_dir = 'data/horse-or-human/train/horses'
    # Directory with our training human pictures
    train_human_dir = 'data/horse-or-human/train/humans'

    print('total training horse images:', len(os.listdir(train_horse_dir)))
    print('total training human images:', len(os.listdir(train_human_dir)))

    train_horse_names = os.listdir(train_horse_dir)
    train_human_names = os.listdir(train_human_dir)
    # plot_images(train_horse_dir, train_human_dir, train_horse_names, train_human_names)

    training_dir = Path('data/horse-or-human/train')

    # # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(
      training_dir,
      target_size=(300, 300),
      class_mode='binary'
    )

    # '''
    # “First, the images are much larger—300 × 300 pixels—so more layers may be needed. 
    # Second, the images are full color, not grayscale, so each image 
    # will have three channels instead of one. Third, there are only two image types, 
    # so we have a binary classifier that can be implemented using just a single output neuron, 
    # where it approaches 0 for one class and 1 for the other. Keep these considerations in mind 
    # when exploring this architecture:”
    # '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    validation_dir = Path('data/horse-or-human/validation')
    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow_from_directory(
      validation_dir,
      target_size=(300, 300),
      class_mode='binary'
    )


    history = model.fit_generator(
      train_generator,
      epochs=6,
      validation_data=validation_generator
    )

    # Print the accuracy on the validation data
    print("Validation Accuracy:", history.history['val_accuracy'])
