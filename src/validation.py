import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the dictionary mapping emotions to labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define data generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the weights of the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Load weights into the model
model.load_weights('src/model.h5')

# Define the directory containing the validation images
val_dir = 'src/data/test'

# Create the validation generator
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False  # Set shuffle to False to keep track of predictions
)

# Perform predictions on the validation dataset
predictions = model.predict(validation_generator)

# Process the predictions and display the results
for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    print("Prediction for image {}: {}".format(i+1, emotion_dict[predicted_class]))

# Calculate accuracy
accuracy = model.evaluate(validation_generator)
print("Validation Accuracy: {:.2f}%".format(accuracy[1] * 100))
