import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,               # The 1D convolutional layer
    MaxPooling1D,         # The 1D max pooling layer
    BatchNormalization,   # The layer for stabilizing learning
    Dropout,              # The layer for preventing overfitting
    Flatten,              # The layer to bridge conv blocks and dense layers
    Dense                 # The classic fully-connected neural network layer
)

#import keras 
#print(keras.__version__)

CSV_PATH = "features.csv"

try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset load successfully")

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    exit()

x=features_df.drop('genre_label',axis=1)
y=features_df['genre_label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

print("\nData preparation complete. Shapes before reshaping:")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")

print("\n--- Reshaping data for CNN model ---")

x_train_cnn = np.expand_dims(X_train_scaled,axis=-1)
x_test_cnn = np.expand_dims(X_test_scaled,axis=-1)

print("\nShapes after reshaping for CNN:")
print(f"X_train_cnn shape: {x_train_cnn.shape}")
print(f"X_test_cnn shape: {x_test_cnn.shape}")


print(f"\ny_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

model = Sequential()
print("Sequential model canvas created successfully")

model.add(Conv1D(filters=32,kernel_size=3,activation='relu',input_shape=(x_train_cnn.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(units=10 , activation='softmax'))

print("\n--- Compiling the CNN Model ---")

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("Model compiled successfully. It is now ready to be trained")

print("\n--- Starting Model Training ---")
history = model.fit(x_train_cnn, y_train, epochs=50, batch_size=32,validation_split=0.2)
print("\n--- Model Training Complete ---")


print("\n--- Plotting Training and Validation History ---")
import matplotlib.pyplot as plt

def plot_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(history.history['accuracy'],label='Training Accuracy')
    axs[0].plot(history.history['val_accuracy'],label='Validation Accuracy')
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].legend(loc="lower right")

    axs[1].plot(history.history["loss"], label="Training Loss")
    # Access the validation loss history
    axs[1].plot(history.history["val_loss"], label="Validation Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_title("Training and Validation Loss")
    axs[1].legend(loc="upper right")
    #Accuracy plot:Blue (training) and orange (validation) should rise together. If orange flattens while blue keeps rising, that’s where overfitting starts.
    #Loss plot:Look for the “elbow” where orange (validation loss) is lowest before rising — that’s the optimal epoch.
    plt.tight_layout()
    plt.show()
plot_history(history)

print("\n--- Saving the trained CNN model to disk ---")
model.save('music_genre_cnn.h5')
print("\nModel successfully saved as 'music_genre_cnn.h5' in your project directory.")
print("This file can now be loaded for future evaluation or deployment.")

print("\n--- Model summary after adding All Convolutional Blocks ---")
model.summary()