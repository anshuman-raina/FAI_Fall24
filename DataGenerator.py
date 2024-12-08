import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os
import apply_random_rotation,apply_random_shearing from '/Utils/data_augmentation_utils'

class DataGenerator(Sequence):

    def __init__(self, dataframe, image_folder, batch_size=16, target_size=(640, 640), augment=False, shuffle=False):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle

        if self.augment:
            self.dataframe = self.augment_dataset()

        self.on_epoch_end()

    def augment_dataset(self):
        augmented_data = []
        os.makedirs(self.output_folder, exist_ok=True)

        for index, row in self.dataframe.iterrows():
            image_path = os.path.join(self.image_folder, row['image_name'])
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image at {image_path}")

            original_height, original_width = image.shape[:2]


            bbox = [
                int(row['bbox_x']),
                int(row['bbox_y']),
                int(row['bbox_x'] + row['bbox_width']),
                int(row['bbox_y'] + row['bbox_height'])
            ]

            for i in range(self.augmentations_per_image):
                augmented_image = image.copy()
                augmented_bbox = bbox.copy()

                # Apply random rotation
                if np.random.rand() < 0.5:
                    augmented_image, augmented_bbox = apply_random_rotation(augmented_image, augmented_bbox)

                # Apply random shearing
                if np.random.rand() < 0.5:
                    augmented_image, augmented_bbox = apply_random_shearing(augmented_image, augmented_bbox)

                # Save augmented image
                augmented_image_name = f"{row['image_name'].split('.')[0]}_aug_{i}.jpg"
                augmented_image_path = os.path.join(self.output_folder, augmented_image_name)
                cv2.imwrite(augmented_image_path, augmented_image)

                # Normalize bounding box and update dataframe
                augmented_data.append({
                    'image_name': augmented_image_name,
                    'bbox_x': augmented_bbox[0],
                    'bbox_y': augmented_bbox[1],
                    'bbox_width': augmented_bbox[2] - augmented_bbox[0],
                    'bbox_height': augmented_bbox[3] - augmented_bbox[1],
                    'label_name': row['label_name']
                })

        augmented_dataframe = pd.concat([self.dataframe, pd.DataFrame(augmented_data)], ignore_index=True)
        return augmented_dataframe


    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataframe))
        batch_indices = self.indices[start_idx:end_idx]

        X, y = self.__data_generation(batch_indices)

        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    
    def __data_generation(self, batch_indices, draw_bboxes=True, output_folder="annotated_images"):


   

        batch_size = len(batch_indices)
        X = np.zeros((batch_size, *self.target_size, 1), dtype=np.float32)
        y_class = np.zeros((batch_size,), dtype=np.int32)  # Integer labels for classification
        y_bbox = np.zeros((batch_size, 4), dtype=np.float32)  # Bounding box coordinates

        if draw_bboxes:
            os.makedirs(output_folder, exist_ok=True)

        for i, idx in enumerate(batch_indices):
            row = self.dataframe.iloc[idx]
            image_path = os.path.join(self.image_folder, row['image_name'])

            
            image, bbox = self.preprocess_image(image_path, row)
            X[i] = image

            label_map = {'empty-shelf': 1}

            y_class[i] = label_map[row['label_name']]  # Map label name to integer

            y_bbox[i] = bbox

            if draw_bboxes:
                annotated_image = self.draw_bounding_box(image, bbox, row['label_name'])
                output_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, annotated_image)

        return X, {'classification_output': y_class, 'bbox_output': y_bbox}

        
    def preprocess_image(self, image_path, row, image_width=640, image_height=640):
        # Read image 
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)

        # Normalize image (0-1 range)
        image = image.astype(np.float32) / 255.0
        
        # Noise reduction using median blur
        image = cv2.medianBlur(image, 3)  # 3x3 kernel
        
        # Normalize bounding box coordinates
        bbox = [
            row['x_top_left'] / image_width,
            row['y_top_left'] / image_height,
            row['x_bottom_right'] / image_width,
            row['y_bottom_right'] / image_height
        ]
        image = np.expand_dims(image, axis=-1)

        return image, np.array(bbox)

    def draw_bounding_box(self, image, bbox, label_name):
        """
        Draw a bounding box with a label on an image.

        Parameters:
            image (np.ndarray): The image on which to draw.
            bbox (np.ndarray): Bounding box in [x_min, y_min, x_max, y_max] format.
            label_name (str): Label for the bounding box.

        Returns:
            annotated_image (np.ndarray): The image with the bounding box and label drawn.
        """
        annotated_image = image.copy()
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, bbox)

        # Determine box and text color
        if label_name.strip() == "empty-shelf":
            box_color = (0, 255, 255)  # Yellow for empty-shelf
            text_color = (0, 255, 255)
        else:
            box_color = (0, 0, 255)  # Red for product
            text_color = (0, 0, 255)

        # Draw rectangle and add text
        cv2.rectangle(annotated_image, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), box_color, 2)
        cv2.putText(
            annotated_image,
            label_name,
            (x_top_left, y_top_left - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )

        return annotated_image

    def draw_bounding_box(self, image, bbox, label_name):
        """
        Draw a bounding box with a label on an image.

        Parameters:
            image (np.ndarray): The image on which to draw.
            bbox (np.ndarray): Bounding box in [x_min, y_min, x_max, y_max] format.
            label_name (str): Label for the bounding box.

        Returns:
            annotated_image (np.ndarray): The image with the bounding box and label drawn.
        """
        annotated_image = image.copy()
        x_top_left, y_top_left, x_bottom_right, y_bottom_right = map(int, bbox)

        # Determine box and text color
        if label_name.strip() == "empty-shelf":
            box_color = (0, 255, 255)  # Yellow for empty-shelf
            text_color = (0, 255, 255)
        else:
            box_color = (0, 0, 255)  # Red for product
            text_color = (0, 0, 255)

        # Draw rectangle and add text
        cv2.rectangle(annotated_image, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), box_color, 2)
        cv2.putText(
            annotated_image,
            label_name,
            (x_top_left, y_top_left - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1
        )

        return annotated_image