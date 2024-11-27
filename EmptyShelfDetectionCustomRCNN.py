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
    label_map = {0: 'empty-shelf', 1: 'product'}
    label_name = label_map[class_index]

    # Denormalize bbox predictions (assuming they're in [0, 1] range)
    x_top_left = bbox_pred[0, 0]
    y_top_left = bbox_pred[0, 1]
    x_bottom_right = bbox_pred[0, 2]
    y_bottom_right = bbox_pred[0, 3]

    # Draw predictions only if box coordinates make sense
    if x_top_left < x_bottom_right and y_bottom_right < y_top_left:
        cv2.rectangle(image, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)
        cv2.putText(image, f'{label_name} ({class_probs[class_index]:.2f})',
                    (x_top_left, y_top_left + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        print("Warning: Invalid box coordinates")

    return image


if __name__ == '__main__':
    main()
