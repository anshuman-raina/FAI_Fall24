import numpy as np
import math
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os

def apply_random_rotation(image, bbox, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    rad_angle = math.radians(angle)
    
    image_center = (image.shape[1] / 2, image.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    rotated_points = rotation_matrix.dot(points_ones.T).T
    
    x_min_rot, y_min_rot = rotated_points[:, 0].min(), rotated_points[:, 1].min()
    x_max_rot, y_max_rot = rotated_points[:, 0].max(), rotated_points[:, 1].max()
    rotated_bbox = [x_min_rot, y_min_rot, x_max_rot, y_max_rot]
    
    return rotated_image, rotated_bbox

def apply_random_shearing(image, bbox, max_shear=15):
    shear_x = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_y = np.random.uniform(-math.radians(max_shear), math.radians(max_shear))
    shear_matrix = np.array([[1, math.tan(shear_x), 0], [math.tan(shear_y), 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
    
    x_min, y_min, x_max, y_max = bbox
    box_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    points_ones = np.hstack([box_points, np.ones((4, 1))])
    sheared_points = shear_matrix.dot(points_ones.T).T
    
    x_min_shear, y_min_shear = sheared_points[:, 0].min(), sheared_points[:, 1].min()
    x_max_shear, y_max_shear = sheared_points[:, 0].max(), sheared_points[:, 1].max()
    sheared_bbox = [x_min_shear, y_min_shear, x_max_shear, y_max_shear]
    
    return sheared_image, sheared_bbox

class DataGenerator(Sequence):
    def __init__(self, dataframe, image_folder, batch_size=16, target_size=(640, 640), augment=True, shuffle=True):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()
    
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
    
    def __data_generation(self, batch_indices):
        batch_size = len(batch_indices)
        X = np.zeros((batch_size, *self.target_size, 3), dtype=np.float32)
        y_class = np.zeros((batch_size,), dtype=np.int32)  # Integer labels for classification
        y_bbox = np.zeros((batch_size, 4), dtype=np.float32)  # Bounding box coordinates

        for i, idx in enumerate(batch_indices):
            row = self.dataframe.iloc[idx]
            image_path = os.path.join(self.image_folder, row['image_name'])
            
            # Load and preprocess image
            image, bbox = self.preprocess_image(image_path, row)
            X[i] = image

            # Assign class label
            label_map = {'empty-shelf': 0, 'product': 1}
            y_class[i] = label_map[row['label_name']]  # Map label name to integer

            # Assign bounding box
            y_bbox[i] = bbox

        return X, {'classification_output': y_class, 'bbox_output': y_bbox}
        
    def preprocess_image(self, image_path, row):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        original_height, original_width = image.shape[:2]
        
        # Normalize bbox coordinates
        x_min = row['bbox_x'] / original_width
        y_min = row['bbox_y'] / original_height
        x_max = (row['bbox_x'] + row['bbox_width']) / original_width
        y_max = (row['bbox_y'] + row['bbox_height']) / original_height
        bbox = [x_min, y_min, x_max, y_max]
        
        # Resize and normalize image
        image = cv2.resize(image, self.target_size)
        image = image.astype(np.float32) / 255.0
        
        return image, np.array(bbox)