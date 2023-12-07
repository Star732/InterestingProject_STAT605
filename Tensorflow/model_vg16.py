import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
import glob
import sys

from tensorflow.keras import datasets, layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tqdm import tqdm
import tensorflow.keras.backend as K

from dataProcess import DataProcessing

### gpu
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]],"GPU")
    
print(gpus)


#### data process
batch_size = 64   ### could be modified
img_height, img_width = 256, 256

train_dir = 'Train'
val_dir = 'Valid'
train_dir = 'Test'

train_ds = DataProcessing(train_dir, batch_size)
val_ds = DataProcessing(val_dir, batch_size)

### check if process has errors
class_names = train_ds.class_names
print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
def preprocess_image(image,label):
    return (image/256., label)

train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

####################
### model - vg16 ###
####################

def VGG16(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv2D(64, (3,3), activation='relu', padding='same',name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3,3), activation='relu', padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'block1_pool')(x)
    # 2nd block
    x = Conv2D(128, (3,3), activation='relu', padding='same',name='block2_conv1')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'block2_pool')(x)
    # 3rd block
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='block3_conv1')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='block3_conv2')(x)
    x = Conv2D(256, (3,3), activation='relu', padding='same',name='block3_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'block3_pool')(x)
    # 4th block
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block4_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block4_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block4_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'block4_pool')(x)
    # 5th block
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block5_conv1')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block5_conv2')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same',name='block5_conv3')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name = 'block5_pool')(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation='relu',  name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(input_tensor, output_tensor)
    return model

model=VGG16(1000, (img_width, img_height, 3))
print(model.summary())

model.compile(optimizer="adam",
              loss     ='sparse_categorical_crossentropy',
              metrics  =['accuracy'])

epochs = 5
lr     = 1e-4

history_train_loss     = []
history_train_accuracy = []
history_val_loss       = []
history_val_accuracy   = []

for epoch in range(epochs):
    train_total = len(train_ds)
    val_total   = len(val_ds)
    
    """
    total：预期的迭代数目
    ncols：控制进度条宽度
    mininterval：进度更新最小间隔，以秒为单位（默认值：0.1）
    """
    with tqdm(total=train_total, desc=f'Epoch {epoch + 1}/{epochs}',mininterval=1,ncols=100) as pbar:
        
        lr = lr*0.92
        K.set_value(model.optimizer.lr, lr)
        
        for image,label in train_ds:      
            """
            train_on_batch is more advanced than model.fit()
            """
            history = model.train_on_batch(image,label)
            
            train_loss     = history[0]
            train_accuracy = history[1]
            
            pbar.set_postfix({"loss": "%.4f"%train_loss,
                              "accuracy":"%.4f"%train_accuracy,
                              "lr": K.get_value(model.optimizer.lr)})
            pbar.update(1)
        history_train_loss.append(train_loss)
        history_train_accuracy.append(train_accuracy)
    
    with tqdm(total=val_total, desc=f'Epoch {epoch + 1}/{epochs}',mininterval=0.3,ncols=100) as pbar:

        for image,label in val_ds:      
            
            history = model.test_on_batch(image,label)
            
            val_loss     = history[0]
            val_accuracy = history[1]
            
            pbar.set_postfix({"loss": "%.4f"%val_loss,
                              "accuracy":"%.4f"%val_accuracy})
            pbar.update(1)
        history_val_loss.append(val_loss)
        history_val_accuracy.append(val_accuracy)
            
    print("loss: %.4f"%val_loss)
    print("accuracy: %.4f"%val_accuracy)


epochs_range = range(epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, history_train_accuracy, label='Training Accuracy')
plt.plot(epochs_range, history_val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_train_loss, label='Training Loss')
plt.plot(epochs_range, history_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('Accuracy_Loss_vg16.pdf')

# ## save model
history.save('model_vg16.h5')

plt.figure(figsize=(18, 12))  # width * height

for images, labels in val_ds:
    for i in range(24):
        ax = plt.subplot(4,6, i + 1)  
        
        plt.imshow(images[i].numpy())
        
        img_array = tf.expand_dims(images[i], 0) 
        
        predictions = model.predict(img_array)
        plt.title(class_names[np.argmax(predictions)])

        plt.axis("off")
    plt.savefig('Result_cnn.pdf')

