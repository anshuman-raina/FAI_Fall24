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

def predict_on_image(image_path, model, target_size=(640, 640)):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    
    # No resizing or normalization
    resized_image = image.copy()
    resized_image = np.expand_dims(resized_image, axis=0)
    
    # Make prediction
    bbox_pred, class_pred = model.predict(resized_image)
 
    # Process predictions
    class_index = np.argmax(class_pred[0])
    class_probs = class_pred[0]
    
    # Map class index to label
    label_map =  {0: 'empty-shelf', 1: 'product'}
    label_name = label_map[class_index]
    
    # Denormalize bbox predictions (assuming they're in [0, 1] range)
    x_min = int(bbox_pred[0, 0] * original_width)
    y_min = int(bbox_pred[0, 1] * original_height)
    x_max = int(bbox_pred[0, 2] * original_width)
    y_max = int(bbox_pred[0, 3] * original_height)

    
    # Draw predictions only if box coordinates make sense
    if x_min < x_max and y_min < y_max:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'{label_name} ({class_probs[class_index]:.2f})', 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("Warning: Invalid box coordinates")
    
    return image

if __name__ == '__main__':
    main()