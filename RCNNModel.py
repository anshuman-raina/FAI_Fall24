from tensorflow.keras import layers, Input
import tensorflow as tf

def build_rcnn_model(input_shape=(640, 640, 3)):
    # Input layer
    input_tensor = Input(shape=input_shape)
    
    # First Convolutional Block
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 320x320
    
    # Second Convolutional Block
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 160x160
    
    # Third Convolutional Block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 80x80
    
    # Fourth Convolutional Block
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 40x40
    
    # Fifth Convolutional Block
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # Downsample to 20x20
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)  # Convert feature maps to a single vector
    
    # Fully Connected Layers
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Classification Output (Binary: Empty or Not Empty)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    
    # Optional: Bounding Box Regression Output
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    # Create Model
    model = tf.keras.Model(inputs=input_tensor, outputs=[classification_output, bbox_output])
    
    return model
