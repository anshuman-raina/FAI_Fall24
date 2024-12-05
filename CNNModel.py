from tensorflow.keras import layers, Input
import tensorflow as tf

def build_cnn_model_with_residuals(input_shape=(640, 640, 1)):
    # Input layer
    input_tensor = Input(shape=input_shape)
    
    # First Convolutional Block
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 320x320
    
    # Second Convolutional Block with Residual Connection
    shortcut = x  # Save input as shortcut
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 160x160
    shortcut = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(shortcut)  # Match dimensions
    x = layers.Add()([x, shortcut])  # Add the shortcut connection
    
    # Third Convolutional Block with Residual Connection
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 80x80
    shortcut = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    # Fourth Convolutional Block with Residual Connection
    shortcut = x
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 40x40
    shortcut = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    # Fifth Convolutional Block with Residual Connection
    shortcut = x
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 20x20
    shortcut = layers.Conv2D(512, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected Layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification Output (Binary: Empty or Not Empty)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    
    # Optional: Bounding Box Regression Output
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    # Create Model
    model = tf.keras.Model(inputs=input_tensor, outputs=[bbox_output, classification_output])
    
    return model
