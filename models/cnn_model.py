from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


def build_model(num_classes):

    model = Sequential()

    # Block 1
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 2
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 3
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 4
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model