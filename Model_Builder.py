import argparse
import tensorflow as tf
from DataGenerator import DataGenerator
from CNNModel import build_cnn_model_with_residuals

from Utils.converter import parse_yolov5_obb


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

    model = build_cnn_model_with_residuals(input_shape=(640, 640,1))

    model.compile(
        optimizer='adam',
        loss={
            'classification_output': 'binary_crossentropy',
            'bbox_output': 'mean_squared_error',
        },
        loss_weights={
            'classification_output': 0.2,
            'bbox_output': 2.0,
        },
        metrics={
            'classification_output': 'accuracy',
            'bbox_output': ['mse', tf.keras.metrics.MeanAbsoluteError(name='mae')],
        },
    )

    model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=len(train_generator),
    )

    # Save model
    model.save('empty_shelf_detector.h5')

if __name__ == '__main__':
    main()

