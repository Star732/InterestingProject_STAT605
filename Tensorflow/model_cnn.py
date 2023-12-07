### import PIL
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

# if len(sys.argv) != 2:
#     print('Usage: %s DATA' % (os.path.basename(sys.argv[0])))
#     sys.exit(1)
# modelNum = sys.argv[0]


### gpu
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]],"GPU")
    
print(gpus)


#### data process
batch_size = 64   ### could be modified

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

### Model - CNN

num_classes = len(class_names)
img_height, img_width = 256, 256

model = models.Sequential([  
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)), 
    layers.MaxPooling2D((2, 2)),                   
    layers.Conv2D(32, (3, 3), activation='relu'),  
    layers.MaxPooling2D((2, 2)),                   
    layers.Conv2D(64, (3, 3), activation='relu'),  
    
    layers.Flatten(),                       
    layers.Dense(128, activation='relu'),   
    layers.Dense(num_classes)               
])

print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# modelName = 'model_' + str(1) + '.h5'
history.save('model_cnn.h5')

### accuracy + loss

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# figname = 'Accuracy_Model' + str(modelNum) + '.pdf'
plt.savefig('Accuracy_Loss_cnn.pdf')

### predict

plt.figure(figsize=(18, 12))

test_dir = 'Test'
test_ds = DataProcessing(test_dir, batch_size)

for images, labels in test_ds.take(1):
    for i in range(24):
        ax = plt.subplot(4,6, i + 1)  

        plt.imshow(images[i].numpy())
        
        img_array = tf.expand_dims(images[i], 0) 
        
        predictions = model.predict(img_array)
        plt.title(class_names[np.argmax(predictions)])

        plt.axis("off")

    # resultName = 'Result_' + str(modelNum) + '.pdf'
    plt.savefig('Result_cnn.pdf')