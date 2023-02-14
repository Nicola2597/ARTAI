import tensorflow as tf
import matplotlib
import tqdm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import efficientnet.keras as efn
import pandas as pd
import os
import efficientnet.tfkeras
from keras.models import load_model
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
from random import shuffle
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image
import plotly.express as px
from keras.preprocessing.image import  ImageDataGenerator as IDG
import urllib.request
from keras import Sequential
from functools import partial
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.utils import to_categorical
#fixed parameters
#https://www.kaggle.com/code/cdeotte/tfrecord-experiments-upsample-and-coarse-dropout/notebook
#https://www.kaggle.com/code/steubk/wikiart-top25-artists-classifier-baseline
channels=3
IMG_SIZE=224
#with the efficientnetB0 i need less then 20 as batch size or it will print me out of memory error
batch_size=12
path=r'C:\Users\nicol\Documents\archive1'
def create_train_test_val(path):
 df=pd.read_csv(os.path.join(path+'/','classes.csv'))
 #drop all non existing path
 df=df[[os.path.isfile(os.path.join(path,i)) for i in df['filename']]]
 #drop the useless columns
 df=df.drop(['phash','width','height','genre_count','description','artist'],axis=1)
 #if the genre column has more than one string then cut it
 df['genre'] = df['genre'].str.split().str[0]
 #remove the brackets from the genre column
 df['genre'] = df['genre'].str.replace('\W', '', regex=True)
 # let's encode the labels
 le = LabelEncoder()
 df["genre_class"] = le.fit_transform(df["genre"].values)
 df = df.sample(frac=1).reset_index(drop=True)
 df_train = df.query("subset == 'train'").reset_index(drop=True)
 #add some more shuffle
 df_train=df_train.sample(frac=1).reset_index(drop=True)
 df_test = df.query("subset == 'test'").reset_index(drop=True)
 df_test = df_test.sample(frac=1).reset_index(drop=True)
 df_train, df_valid, y_train, y_valid = train_test_split(df_train, df_train["genre_class"],
                                                        test_size=0.2, random_state=42,
                                                        stratify=df_train["genre_class"])
 y_test=df_test["genre_class"]
 df_test=df_test.drop(['genre','subset'],axis=1)
 return df_train,df_valid,df_test,y_train,y_valid,y_test


def dropout(img, size, p, count, cutout_masksize):
    # input image - is one image of size [size,size,3] not a batch of [b,size,size,3]
    # output - image with <count> squares of side <size*cutout_masksize> removed

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    #calcola prima la random uniform tra 0 e 1 e se è >di p allora diventa 1 ed essendo 1 p=1 mentre se è<p diventa zero quindi p esce o 0 o 1
    p = tf.cast(tf.random.uniform([], 0, 1)<p , 'int32')

    if (p == 0) | (count == 0) | (cutout_masksize == 0): return img

    for k in range(count):

        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, size), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, size), tf.int32)
        # COMPUTE SQUARE
        width = tf.cast(cutout_masksize * size, tf.int32)

        ya = tf.math.maximum(0, y - width // 2)
        yb = tf.math.minimum(size, y + width // 2)
        xa = tf.math.maximum(0, x - width // 2)
        xb = tf.math.minimum(size, x + width // 2)
        # DROPOUT IMAGE
        one = img[ya:yb, 0:xa, :]
        two = tf.zeros([yb - ya, xb - xa, 3])
        three = img[ya:yb, xb:size, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([img[0:ya, :, :], middle, img[yb:size, :, :]], axis=0)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR
    img = tf.reshape(img, [size, size, 3])

    return img

def tensorslicedataset(df,path):
 filenames=[f"{path}/{filename}" for filename in df['filename']]
 labels=[label for label in df['genre_class']]
 ds=tf.data.Dataset.from_tensor_slices((filenames, labels))
 ds=ds.shuffle(len(filenames))
 return ds

def parse_function(filename, label):
    image_string = tf.io.read_file(filename)

    #Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    return image, label



def train_preprocess(image, label,show=True):


    cutout_p=0.75
    cutout_count = 5
    cutout_masksize = 0.2
    crop_size = 0.5
    crop_p=0.75
    #quest crop esce 1 o 0
    crop_p = tf.cast( tf.random.uniform([],0,1)<crop_p, tf.float32)

    r = 1.0 + crop_size *np.random.random()* crop_p

    a=int(IMG_SIZE * r)
    image = tf.image.resize(image, size=[a,a])

    image = tf.image.random_crop(
        image, size=[IMG_SIZE, IMG_SIZE, 3]
    )
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_contrast(image,lower= 0.8, upper=1.2)
    image = dropout(image, size=IMG_SIZE, p=cutout_p, count=cutout_count, cutout_masksize=cutout_masksize)

    if show:image = tf.cast(image,tf.float32)/255

    return image, label
def test_preprocess(image, label):
    image = tf.cast(image,tf.float32)/255

    return image, label

def prefetchDataset(df,train=False):
 dataset=tensorslicedataset(df,path)
 dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
 if train:
     dataset = dataset.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
 else:
     dataset = dataset.map(test_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
 dataset = dataset.batch(batch_size)
 dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
 return dataset

df_train,df_valid,df_test,y_train,y_valid,y_test=create_train_test_val(path)
train_ds=prefetchDataset(df_train,True)
valid_ds=prefetchDataset(df_valid)
test_ds=prefetchDataset(df_test)
#use the gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# ResNet model neural network
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

def show_train(df_train):
    df = df_train.sample(n=20, random_state=42)
    filenames = [f"{path}/{filename}" for filename in df["filename"].values]
    labels = df["genre"].values

    i = 1
    plt.figure(figsize=(15, 20))

    for images, label in zip(filenames,labels):
        image_string = tf.io.read_file(images)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_jpeg(image_string, channels=3)
        image,label=train_preprocess(image,label,show=False)

        plt.subplot(5, 4, i)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(label)
        plt.axis('off')

        i += 1
        if i >= 21: break

    plt.show()
#show 20 random images from the train folder
#show_train(df_train)

#Learning Rate Annealer
lrr= ReduceLROnPlateau(   monitor='val_accuracy',   factor=.01,   patience=2,  min_lr=1e-5)

#Defining the model
base_model = efn.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE,IMG_SIZE,channels))
model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(27, activation='sigmoid')
    ])
# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'])
cbs = [
    EarlyStopping(patience=8, restore_best_weights=True),
    ModelCheckpoint('test1weather' + ".h5",model=['val_accuracy','val_loss'], save_best_only=True)
]

history = model.fit(train_ds, epochs=100, verbose=1, validation_data=valid_ds, callbacks=[lrr,cbs],use_multiprocessing=True,steps_per_epoch = len(train_ds),validation_steps = len(valid_ds))
