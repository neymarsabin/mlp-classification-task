import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.python.module.module import valid_identifier

DATASET_DIR = "./datasets/"
BASE_DIR = "./"

# configuration
batch_size = 55
image_height = 128
image_width = 128
image_size = (image_height, image_width)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
print("Classes for MLP classification:", class_names)

# Cache & prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# rescale images to [0, 1]
normalization_layer = tf.keras.layers.Rescaling(1./255)

def preprocess(ds):
    return ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = preprocess(train_ds)
val_ds = preprocess(val_ds)

## let's define the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(image_height, image_width, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

model.save(BASE_DIR + "classification-model.h5")

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
