from keras.applications import ResNet50V2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

ResNet50V2_model = ResNet50V2(weights="imagenet",
                                include_top=False,
                                input_shape=(150, 150, 3))

ResNet50V2_model.trainable = False

model = Sequential()
model.add(ResNet50V2_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])

train_dir = 'Data/train'
val_dir = 'Data/validation'
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)
epochs = 10
batch_size = 10
nb_train_samples = 100
nb_validation_samples = 50

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(directory=train_dir,
                                              target_size=(img_width,
                                                           img_height),
                                              batch_size=batch_size,
                                              class_mode='binary')
val_generator = datagen.flow_from_directory(directory=val_dir,
                                            target_size=(img_width,
                                                         img_height),
                                            batch_size=batch_size,
                                            class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator, 
    validation_steps = val_generator.samples // batch_size,
    epochs = epochs)

model.save('resnetАВ.h5')