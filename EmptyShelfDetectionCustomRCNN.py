import pandas as pd
import argparse
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from DataAugmentationHelper import DataGenerator
from RCNNModel import build_rcnn_model

from converter import parse_yolov5_obb

def main():
    parser = argparse.ArgumentParser(description='Empty Shelf Detection Training')
    parser.add_argument('--annotations_dir', required=True, help='Path to the YOLOv5-OBB annotations directory')
    parser.add_argument('--image_folder', required=True, help='Path to the folder containing images')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    args = parser.parse_args()
    
    # Load data
    data = parse_yolov5_obb(args.annotations_dir, args.image_folder)
    
    # Initialize data generator
    train_generator = DataGenerator(
        dataframe=data,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        target_size=(640, 640),
    )
    
    # Build and compile model
    tf.keras.backend.clear_session()

    model = build_rcnn_model(input_shape=(640, 640, 3))

    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'binary_crossentropy',
            'bbox_output': 'mean_squared_error',
        },
        loss_weights={
            'classification_output': 1.0,
            'bbox_output': 1.5,
        },
        metrics={
            'classification_output': 'accuracy',
            'bbox_output': 'mse',
        },
    )

    model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=len(train_generator),
    )
    
    # Save model
    model.save('empty_shelf_detector_rcnn_resnet.h5')

def predict_on_image(image_path, model, label_map={0: 'empty-shelf', 1: 'product'}):
    # Read image and get original dimensions
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    # Prepare image for prediction
    resized_image = image.copy()
    resized_image = np.expand_dims(resized_image, axis=0)
    
    # Make prediction
    bbox_pred, class_pred = model.predict(resized_image)
 
    # Process predictions
    class_index = np.argmax(class_pred[0])
    class_probs = class_pred[0]
    
    # Get label name
    label_name = label_map.get(class_index, 'Unknown')
    
    # Extract top-left and bottom-right coordinates
    x1, y1 = int(bbox_pred[0, 0] * original_width), int(bbox_pred[0, 1] * original_height)
    x2, y2 = int(bbox_pred[0, 2] * original_width), int(bbox_pred[0, 3] * original_height)
    
    # Validate and clamp coordinates
    x1 = max(0, min(x1, original_width))
    y1 = max(0, min(y1, original_height))
    x2 = max(0, min(x2, original_width))
    y2 = max(0, min(y2, original_height))
    
    # Draw if coordinates are valid
    if x1 < x2 and y1 < y2:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        confidence = class_probs[class_index]
        cv2.putText(image, f'{label_name} ({confidence:.2f})', 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image