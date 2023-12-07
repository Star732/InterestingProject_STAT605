from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import glob

img_height = 256
img_width = 256

def DataProcessing(dir, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dir,  ## directory
        validation_split= 0.,
        image_size=(img_height, img_width),
        batch_size=batch_size
        )
    
    return train_ds
