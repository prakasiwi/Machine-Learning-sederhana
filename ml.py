##### Mengimport file-file yang dibutuhkan
import tensorflow as tf                                                           # import tensorflow
import keras_preprocessing                                                        # import keras                                             
from keras_preprocessing.image import ImageDataGenerator                          # import Image Data Generator

print(tf.__version__)

##### Mengunduh dataset yang telah disediakan 

!wget --no-check-certificate \
https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip

##### Ekstrasi file zip

# ekstrasi file zip
import zipfile

local_zip = '/content/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

##### Mendefinisakan folder yang akan digunakan

import os

gambar_gunting = os.path.join('/content/rockpaperscissors/scissors')
gambar_batu = os.path.join('/content/rockpaperscissors/rock')
gambar_kertas = os.path.join('/content/rockpaperscissors/paper')

##### Melihat jumlah gambar pada setiap folder

print("Banyak gambar gunting :", len(os.listdir(gambar_gunting)))
print("Banyak gambar batu :", len(os.listdir(gambar_batu)))
print("Banyak gambar kertas :", len(os.listdir(gambar_kertas)))

##### Melakukan data generator

base_dir = '/content/rockpaperscissors/rps-cv-images'
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',  
    validation_split=0.4                                                         # membagi validasi sebesar 40%
)

##### Mendefinikan Image Data Generator

train_generator = training_datagen.flow_from_directory(
    base_dir,                                                                     # Folder utama
    target_size=(150, 150),                                                       # Mengubah resolusi seluruh gambar
    class_mode='categorical',                                                     # Mengunakan categorical karena kelas lebih dari 2
    subset='training'                                                             # Untuk training
)
test_generator = training_datagen.flow_from_directory(
    base_dir,                                                                     # Folder utama
    target_size=(150, 150),                                                       # Mengubah resolusi seluruh gambar
    class_mode='categorical',                                                     # Menggunakan categorical karenal kelas lebih dari 2
    subset='validation'                                                           # Untuk validasi
)

##### Membangun Jaringan Saraf Tiruan

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)), # Layer konvolusi pertama
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                         # Layer konvolusi kedua
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),                         # Layer konvolusi ketiga
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),                                # Hiden layer 1

    tf.keras.layers.Dense(512, activation='relu'),                                # Hiden layer 2

    tf.keras.layers.Dense(3, activation='softmax'),                               # Output, menggunakan softmax karena disini terdapat multikelas
])
model.summary()

##### Optimizer dan loss-function

model.compile(loss = 'categorical_crossentropy',                                  # Karena kelas lebih dari 2
              optimizer = 'adam',                                                 # Menggunakan optimizer adam yang telah diajarkan
              metrics = ['accuracy'])                                              

##### Melatih model dengan fit

model.fit(
    train_generator,
    steps_per_epoch=25,                                                           
    epochs=20,                                                                    # Menambahkan epoch agar akurasi maksimal
    validation_data=test_generator,                                               # Pengujian di data validasi
    validation_steps=5,                                                           # Batch yang akan dieksekusi setiap epoch
    verbose=2)                                                                    # Semakin rendah loss dan accuracy tinggi maka libih baik

#####

import numpy as np                                                                # mengimport file yang dibutuhkan 
from google.colab import files
from keras_preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

uploaded = files.upload()                                                         # Upload file 

for fn in uploaded.keys():
  
  path = fn
  img = image.load_img(path, target_size=(150,150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
# Proses Percabangan
  print(fn)                                                                       # Gambar akan diprediksi mesin
  if classes[0][0]== 1.0:                                                         # Gambar menghasilkan nilai kertas
    print('Paper')
  elif classes[0][1]==1.0:                                                        # Gambar menghasilkan nilai batu
    print('Rock')
  else:                                                                           # Gambar menghasilkan nilai gunting
    print('Scissors')
