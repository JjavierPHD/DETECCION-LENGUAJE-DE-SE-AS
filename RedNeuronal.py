import cv2
import mediapipe as mp
import os
import tensorflow as tf
from keras.optimizers.adam import optimizer
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K

K.clear_session()

datos_entrenamiento = 'C:/Users/USUARIO/PycharmProjects/Dedos/fotos/entrenamiento'
datos_validacion = 'C:/Users/USUARIO/PycharmProjects/Dedos/fotos/validacion'
iteraciones = 20
altura, longitud = 200, 200
batch_size = 1
pasos = 300 // batch_size
pasos_validacion = 300 // batch_size
filtrosconv1 = 32
filtrosconv2 = 64
filtrosconv3 = 128
tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)
tam_filtro3 = (2, 2)
tam_pool = (2, 2)
clases = 5
lr = 0.0005

preprocesamientos_entre = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)
preprocesamientos_vali = ImageDataGenerator(
    rescale=1./255
)

images_entreno = preprocesamientos_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

images_validacion = preprocesamientos_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn = Sequential()
cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Conv2D(filtrosconv3, tam_filtro3, padding='same'))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Flatten())
cnn.add(Dense(64))
cnn.add(Activation('relu'))
cnn.add(Dense(clases))
cnn.add(Activation('softmax'))

optimizar = tf.keras.optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])
cnn.fit(
    images_entreno,
    steps_per_epoch=pasos,
    epochs=iteraciones,
    validation_data=images_validacion,
    validation_steps=pasos_validacion
)
cnn.save('ModeloVocales.h5')
cnn.save_weights('pesosVocales.h5')
