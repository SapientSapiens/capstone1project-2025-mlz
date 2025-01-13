# importing the libraries 
import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set SEED value
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Best Evaluated Parameters 
learning_rate = 0.001  # Set the best evaluated learning rate
size_inner = 100  # Set the best evaluated inner layer input size
droprate = 0.5  # Set the best evaluated dropout rate

# since we train a larger model (299*299)
input_size =299

# Image data generators
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Image reference loading
train_ds = train_gen.flow_from_directory(
    './dataset/train',
    target_size=(input_size, input_size),
    batch_size=32,
    seed=SEED
)

val_ds = val_gen.flow_from_directory(
    './dataset/val',
    target_size=(input_size, input_size),
    batch_size=32,
    seed=SEED,
    shuffle=False
)

# create the model architechture for the larger model with best parameters
def create_model_architechture(input_size, learning_rate, size_inner, droprate):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(25)(drop)
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model



# Checkpointing to save the best model
checkpoint =  keras.callbacks.ModelCheckpoint(
    'xception_v_script_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Create the model
model = create_model_architechture(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size_inner,
    droprate=droprate
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*HDF5.*")

# Training the larger model
print('Learning rate:', learning_rate)
print('Inner layer size:', size_inner)
print('Dropout rate:', droprate)
print()
print()

history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[checkpoint]
)

print('Training complete.')
