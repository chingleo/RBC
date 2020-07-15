import os,shutil
import glob

import keras

import keras_applications

import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras import layers
from keras import models
from keras import optimizers

from keras_applications.vgg16 import VGG16

import numpy as np

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
#

#original_dataset_dir ='C:/software/Deeplearning/test_ML/kaggle_DogsVSCats/original_data'

base_dir = './temp'
# os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,'lsc')
# os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'tur')
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'lsc')
# os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'tur')
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'lsc')
# os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'tur')
# os.mkdir(test_dogs_dir)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Experiment with your own parameters here to really try to drive it to 99.9% accuracy or better
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Experiment with your own parameters here to really try to drive it to 99.9% accuracy or better
validation_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.

# Note that this may take some time.
history = model.fit(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)

#=================================����====================================
model.save('temperature.h5')
#=================================��ͼ=====================================

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

import time

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------

plt.figure(1)

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.legend()

plt.savefig('accuracy.png')
#plt.close(1)

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.figure(2)

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.legend()

plt.savefig('loss.png')
#plt.close(2)

# Desired output. Charts with training and validation metrics. No crash :)

# predicting images
file_path = './test_imgs/'
f_names = glob.glob(file_path + '*.png')

# 把图片读取出来放到列表中
for i in range(len(f_names)):
    images = image.load_img(f_names[i],target_size=(150,150))
    img_tensor = image.img_to_array(images)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor /= 255.

    print('loading no.%s image' % i)

#    plt.imshow(img_tensor[0])
#    plt.show()

    img = np.vstack([img_tensor])
    classes = model.predict(img, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(i, " is LSC")
    else:
        print(i, " is a chaotic")

######################  plt show  ###################
plt.show()